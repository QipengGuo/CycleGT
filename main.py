import tqdm
import random
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import json
from sklearn.metrics import f1_score
from collections import defaultdict
import copy
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from t2g_model import ModelLSTM
from g2t_model import GraphWriter
from data import write_txt, tensor2data_g2t, tensor2data_t2g, batch2tensor_t2g, batch2tensor_g2t, \
    scan_data, DataPool, fill_pool

import logging
logging.basicConfig(level=logging.INFO)
logging.info('Start Logging')
bleu = Bleu(4)
meteor = Meteor()
rouge = Rouge()
cider = Cider()

def fake_sent(x):
    return ' '.join(['<ENT_{0:}>'.format(xx) for xx in range(len(x))])

def prep_data(config, load=""):
    #prep data always has two steps, build the vocabulary first and then generate data samples
    train_raw = json.load(open(config['train_file'], 'r'))
    max_len = sorted([len(x['text'].split()) for x in train_raw])[int(0.95*len(train_raw))]
    train_raw = [x for x in train_raw if len(x['text'].split())<max_len]
    train_raw = train_raw[:int(len(train_raw)*config['split'])]
    
    dev_raw = json.load(open(config['dev_file'], 'r'))
    test_raw = json.load(open(config['test_file'], 'r'))
    if len(load)==0:
        # scan the data and create vocabulary
        vocab = scan_data(train_raw)
        vocab = scan_data(dev_raw, vocab)
        vocab = scan_data(test_raw, vocab, sp=True)
        for v in vocab.values():
            v.build()
            logging.info('Vocab Size {0:}, detached by test set {1:}'.format(len(v), len(v.sp)))
        return vocab
    else:
        vocab = torch.load(load)['vocab']

    logging.info('MAX_LEN {0:}'.format(max_len))
    pool = DataPool()
    _raw = []
    for x in train_raw:
        _x = copy.deepcopy(x)
        if config['mode']=='sup':
            _raw.append(_x)
        else: #make sure that no information leak in unsupervised settings
            _x['relations'] = []
            _raw.append(_x)

    fill_pool(pool, vocab, _raw, 'train_g2t')
    _raw = []
    for x in train_raw:
        _x = copy.deepcopy(x)
        if config['mode']=='sup':
            _raw.append(_x)
        else: #make sure that no information leak in unsupervised settings
            _x['text'] = fake_sent(_x['entities'])
            _raw.append(_x)

    fill_pool(pool, vocab, _raw, 'train_t2g')
    _raw = []
    for x in dev_raw:
        _x = copy.deepcopy(x)
        _x['text'] = fake_sent(_x['entities'])
        _raw.append(_x)

    fill_pool(pool, vocab, dev_raw, 'dev')
    fill_pool(pool, vocab, _raw, 'dev_t2g_blind') # prepare for the entity2graph setting
    fill_pool(pool, vocab, test_raw, 'test')
    return pool, vocab

def prep_model(config, vocab):
    g2t_model = GraphWriter(copy.deepcopy(config['g2t']), vocab)
    t2g_model = ModelLSTM(relation_types=len(vocab['relation']), d_model=config['t2g']['nhid'], dropout=config['t2g']['drop'])
    return g2t_model, t2g_model

vae_step = 0.
def train_g2t_one_step(batch, model, optimizer, config):
    global vae_step
    model.train()
    optimizer.zero_grad()
    pred, pred_c, kld_loss = model(batch)
    loss = F.nll_loss(pred.reshape(-1, pred.shape[-1]), batch['tgt'].reshape(-1), ignore_index=0)
    loss = loss #+ 1.0 * ((1.-pred_c.sum(1))**2).mean() #coverage penalty
    loss = loss + min(1.0, (vae_step+100)/(vae_step+10000)) * 8.0* 1./385 * kld_loss # magic number 
    vae_step += 1
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
    optimizer.step()
    return loss.item(), kld_loss.item()

def train_t2g_one_step(batch, model, optimizer, config, t2g_weight=None):
    model.train()
    if t2g_weight is not None:
        # category weights 
        weight = torch.from_numpy(t2g_weight).float().to(config['device'])
    optimizer.zero_grad()
    pred = model(batch)
    if t2g_weight is not None:
        loss = F.nll_loss(pred.contiguous().view(-1, pred.shape[-1]), batch['tgt'].contiguous().view(-1), ignore_index=0, weight=weight)
    else:
        loss = F.nll_loss(pred.contiguous().view(-1, pred.shape[-1]), batch['tgt'].contiguous().view(-1), ignore_index=0)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
    optimizer.step()
    return loss.item()

def t2g_teach_g2t_one_step(raw_batch, model_t2g, model_g2t, optimizer, config, vocab):
    # train a g2t model with the synthetic input from t2g model
    model_t2g.eval()
    model_g2t.train()
    batch_t2g = batch2tensor_t2g(raw_batch, config['t2g']['device'], vocab)
    with torch.no_grad():
        t2g_pred = model_t2g(batch_t2g).argmax(-1).cpu()
    syn_batch = []
    for _i, sample in enumerate(t2g_pred):
        rel = []
        for e1 in range(len(raw_batch[_i]['ent_text'])):
            for e2 in range(len(raw_batch[_i]['ent_text'])):
                try:
                    if sample[e1, e2]!=3 and sample[e1, e2]!=0: # 3 is no relation and 0 is <PAD>
                        rel.append([e1, int(sample[e1, e2]), e2])
                except:
                    logging.warn('{0:}'.format([[vocab['entity'](x) for x in y] for y in raw_batch[_i]['ent_text']]))
                    logging.warn('{0:}'.format(sample.size()))
        _syn = tensor2data_t2g(raw_batch[_i], rel, vocab)
        syn_batch.append(_syn) 
    if len(syn_batch)==0:
        return None
    batch_g2t = batch2tensor_g2t(syn_batch, config['g2t']['device'], vocab)
    loss, kld = train_g2t_one_step(batch_g2t, model_g2t, optimizer, config['g2t'])
    return loss, kld

def g2t_teach_t2g_one_step(raw_batch, model_g2t, model_t2g, optimizer, config, vocab, t2g_weight=None):
    # train a t2g model with the synthetic input from g2t model
    model_g2t.eval()
    model_t2g.train()
    syn_batch = []
    if len(raw_batch)>0:
        batch_g2t = batch2tensor_g2t(raw_batch, config['g2t']['device'], vocab)
        with torch.no_grad():
            g2t_pred = model_g2t(batch_g2t, beam_size=1).cpu()
        for _i, sample in enumerate(g2t_pred):
            _s = sample.tolist()
            if 2 in _s: # <EOS> in list
                _s = _s[:_s.index(2)]
            _syn = tensor2data_g2t(raw_batch[_i], _s)
            syn_batch.append(_syn) 
    batch_t2g = batch2tensor_t2g(syn_batch, config['t2g']['device'], vocab, add_inp=True)
    loss = train_t2g_one_step(batch_t2g, model_t2g, optimizer, config['t2g'], t2g_weight=t2g_weight)
    return loss

def eval_g2t(pool, _type, vocab, model, config, display=True):
    logging.info('Eval on {0:}'.format(_type))
    model.eval()
    hyp, ref, _same = [], [], []
    unq_hyp = {}
    unq_ref = defaultdict(list)
    batch_size = 8*config['batch_size']
    with tqdm.tqdm(list(pool.draw_with_type(batch_size, False, _type)), disable=False if display else True) as tqb:
        for i, _batch in enumerate(tqb):
            with torch.no_grad():
                batch = batch2tensor_g2t(_batch, config['device'], vocab)
                seq = model(batch, beam_size=config['beam_size'])
            r = write_txt(batch, batch['tgt'], vocab['text'])
            h = write_txt(batch, seq, vocab['text'])
            _same.extend([str(x['raw_relation'])+str(x['ent_text']) for x in _batch])
            hyp.extend(h)
            ref.extend(r)
        hyp = [x[0] for x in hyp]
        ref = [x[0] for x in ref]
        idxs, _same= list(zip(*sorted(enumerate(_same), key=lambda x:x[1])))

        ptr = 0
        for i in range(len(hyp)):
            if i>0 and _same[i]!=_same[i-1]:
                ptr +=1
            unq_hyp[ptr] = hyp[idxs[i]]
            unq_ref[ptr].append(ref[idxs[i]])
            
        max_len = max([len(ref) for ref in unq_ref.values()])
        unq_hyp = sorted(unq_hyp.items(), key=lambda x:x[0])
        unq_ref = sorted(unq_ref.items(), key=lambda x:x[0])
        hyp = [x[1] for x in unq_hyp]
        ref = [[x.lower() for x in y[1]] for y in unq_ref]

    wf_h = open('hyp.txt', 'w')
    for i,h in enumerate(hyp):
        wf_h.write(str(h)+'\n')
    wf_h.close()
    hyp = dict(zip(range(len(hyp)), [[x.lower()] for x in hyp]))
    ref = dict(zip(range(len(ref)), ref))
    ret = bleu.compute_score(ref, hyp)
    logging.info('BLEU INP {0:}'.format(len(hyp)))
    logging.info('BLEU 1-4 {0:}'.format(ret[0]))
    logging.info('METEOR {0:}'.format(meteor.compute_score(ref, hyp)[0]))
    logging.info('ROUGE_L {0:}'.format(rouge.compute_score(ref, hyp)[0]))
    logging.info('Cider {0:}'.format(cider.compute_score(ref, hyp)[0]))

    return ret[0][-1]

def eval_t2g(pool, _type, vocab, model, config, display=True):
    # evaluate t2g model
    logging.info('Eval on {0:}'.format(_type))
    hyp, ref, pos_label = [], [], []
    model.eval()
    wf = open('t2g_show.txt', 'w')
    with tqdm.tqdm(list(pool.draw_with_type(config['batch_size'], False, _type)), disable=False if display else True) as tqb: 
        for i, _batch in enumerate(tqb):
            with torch.no_grad():
                batch = batch2tensor_t2g(_batch, config['device'], vocab)
                pred = model(batch)
            _pred = pred.view(-1, pred.shape[-1]).argmax(-1).cpu().long().tolist()
            _gold = batch['tgt'].view(-1).cpu().long().tolist()
            tpred = pred.argmax(-1).cpu().numpy()
            tgold = batch['tgt'].cpu().numpy()

            cnts = []
            for j in range(len(_batch)):
                _cnt = 0
                ents = [[y for y in vocab['entity'](x) if y[0]!='<'] for x in _batch[j]['ent_text']]
                wf.write('=====================\n')
                rels = [] 
                for e1 in range(len(ents)):
                    for e2 in range(len(ents)):
                        if tpred[j, e1, e2]!=3 and tpred[j,e1,e2]!=0:
                            rels.append((e1,int(tpred[j,e1,e2]), e2))
                wf.write(str([(ents[e1], vocab['relation'](r), ents[e2]) for e1,r,e2 in rels])+'\n')
                rels = []
                for e1 in range(len(ents)):
                    for e2 in range(len(ents)):
                        if tgold[j, e1, e2]!=3 and tgold[j,e1,e2]!=0:
                            rels.append((e1,int(tgold[j,e1,e2]), e2))
                        if tgold[j,e1,e2]>0:
                            _cnt += 1
                wf.write(str([(ents[e1], vocab['relation'](r), ents[e2]) for e1,r,e2 in rels])+'\n')
                cnts.append(_cnt)
            
            pred, gold = [], []
            for j in range(len(_gold)):
                if _gold[j]>0: # not the <PAD>
                    pred.append(_pred[j])
                    gold.append(_gold[j])
            pos_label.extend([x for x in gold if x!=3]) # 3 is no relation
            hyp.extend(pred)
            ref.extend(gold)
    wf.close()
    pos_label = list(set(pos_label))
    
    f1_micro = f1_score(ref, hyp, average='micro', labels=pos_label, zero_division=0)
    f1_macro = f1_score(ref, hyp, average='macro', labels=pos_label, zero_division=0)

    logging.info('F1 micro {0:} F1 macro {1:}'.format(f1_micro, f1_macro))
    return f1_micro

def warmup_step1(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab):
    model_g2t.blind, model_t2g.blind = True, True
    batch = batch2tensor_t2g(batch_t2g, config['t2g']['device'], vocab)
    loss1 = train_t2g_one_step(batch, model_t2g, optimizerT2G, config['t2g'], t2g_weight=t2g_weight)
    batch = batch2tensor_g2t(batch_g2t, config['g2t']['device'], vocab)
    loss2, kld = train_g2t_one_step(batch, model_g2t, optimizerG2T, config['g2t'])
    return loss1, loss2, kld

def warmup_step2(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab):
    model_g2t.blind, model_t2g.blind = True, False
    _loss1 = g2t_teach_t2g_one_step(batch_t2g, model_g2t, model_t2g, optimizerT2G, config, vocab, t2g_weight=t2g_weight)
    model_g2t.blind, model_t2g.blind = False, True
    _loss2, kld = t2g_teach_g2t_one_step(batch_g2t, model_t2g, model_g2t, optimizerG2T, config, vocab)
    return _loss1, _loss2, kld

def supervise(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab):
    model_g2t.blind, model_t2g.blind = False, False
    batch = batch2tensor_t2g(batch_t2g, config['t2g']['device'], vocab)
    _loss1 = train_t2g_one_step(batch, model_t2g, optimizerT2G, config['t2g'], t2g_weight=t2g_weight)
    batch = batch2tensor_g2t(batch_g2t, config['g2t']['device'], vocab)
    _loss2, kld = train_g2t_one_step(batch, model_g2t, optimizerG2T, config['g2t'])
    return _loss1, _loss2, kld

def back_translation(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab):
    model_g2t.blind, model_t2g.blind = False, False
    _loss1 = g2t_teach_t2g_one_step(batch_t2g, model_g2t, model_t2g, optimizerT2G, config, vocab, t2g_weight=t2g_weight)
    _loss2, kld = t2g_teach_g2t_one_step(batch_g2t, model_t2g, model_g2t, optimizerG2T, config, vocab)
    return _loss1, _loss2, kld

def train(_type, config, load='tmp_vocab.pt'):
    dev_id = 0
    device = torch.device(dev_id)
    config['g2t']['device'] = device
    config['t2g']['device'] = device
    pool, vocab = prep_data(config['main'], load=load)
    model_g2t, model_t2g = prep_model(config, vocab)
    model_g2t.to(device)
    model_t2g.to(device)

    from transformers.optimization import get_cosine_schedule_with_warmup , get_linear_schedule_with_warmup	
    optimizerG2T = torch.optim.Adam(model_g2t.parameters(), lr = config['g2t']['lr'], weight_decay=config['g2t']['weight_decay']) 
    schedulerG2T = get_cosine_schedule_with_warmup(
		optimizer = optimizerG2T , 
		num_warmup_steps = 400 , 
		num_training_steps = 800 * config['main']['epoch'], 
	)
    optimizerT2G = torch.optim.Adam(model_t2g.parameters(), lr = config['t2g']['lr'], weight_decay=config['t2g']['weight_decay'])
    schedulerT2G = get_cosine_schedule_with_warmup(
		optimizer = optimizerT2G , 
		num_warmup_steps = 400 , 
		num_training_steps = 800 * config['main']['epoch'], 
	)
    loss_t2g, loss_g2t = [], []
    best_g2t, best_t2g = 0., 0.
    klds = []

    t2g_weight = [vocab['relation'].wf.get(x, 0) for x in vocab['relation'].i2s]
    t2g_weight[0] = 0
    max_w = max(t2g_weight)
    t2g_weight = np.array(t2g_weight).astype('float32')
    t2g_weight = (max_w+1000)/(t2g_weight+1000)

    for i in range(0, config['main']['epoch']):
        _data_g2t = list(pool.draw_with_type(config['main']['batch_size'], True, _type+'_g2t'))
        _data_t2g = list(pool.draw_with_type(config['main']['batch_size'], True, _type+'_t2g'))

        data_list = list(zip(_data_g2t, _data_t2g))
        _data = data_list
        with tqdm.tqdm(_data, disable=True if not config['main']['display'] else False) as tqb:
            for j, (batch_g2t, batch_t2g) in enumerate(tqb):
                if i<config['main']['pre_epoch'] and config['main']['mode']=='warm_unsup':
                    _loss1, _loss2, kld = warmup_step1(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab)
                if i==config['main']['pre_epoch']+1 and config['main']['mode']=='warm_unsup':
                    _loss1, _loss2, kld = warmup_step2(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab)
                if config['main']['mode']=='sup':
                    _loss1, _loss2, kld = supervise(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab)
                if (i>=config['main']['pre_epoch']+1 and config['main']['mode']=='warm_unsup') or (config['main']['mode']=='cold_unsup'):
                    _loss1, _loss2, kld = back_translation(batch_g2t, batch_t2g, model_g2t, model_t2g, optimizerG2T, optimizerT2G, config, t2g_weight, vocab)
                loss_t2g.append(_loss1)
                schedulerT2G.step()
                loss_g2t.append(_loss2)
                schedulerG2T.step()
                klds.append(kld)
                tqb.set_postfix({'t2g loss': np.mean(loss_t2g), 'g2t loss': np.mean(loss_g2t), 'kld loss': np.mean(klds)})

        logging.info('Epoch '+str(i))
        if i%1==0:
            if i<config['main']['pre_epoch'] and config['main']['mode']=='warm_unsup':
                model_g2t.blind, model_t2g.blind = True, True
            else:
                model_g2t.blind, model_t2g.blind = False, False
            if model_t2g.blind:
                e = eval_t2g(pool, 'dev_t2g_blind', vocab, model_t2g, config['t2g'], display=config['main']['display'])
            else:
                e = eval_t2g(pool, 'dev', vocab, model_t2g, config['t2g'], display=config['main']['display'])
            if e > best_t2g:
                best_t2g = max(best_t2g, e)
                torch.save(model_t2g.state_dict(), config['t2g']['save']+'X'+'best')
            e = eval_g2t(pool, 'dev', vocab, model_g2t, config['g2t'], display=config['main']['display'])
            if e > best_g2t:
                best_g2t = max(best_g2t, e)
                torch.save(model_g2t.state_dict(), config['g2t']['save']+'X'+'best')
            if i==config['main']['pre_epoch']:
                torch.save(model_t2g.state_dict(), config['t2g']['save']+'X'+'mid')
                torch.save(model_g2t.state_dict(), config['g2t']['save']+'X'+'mid')
    model_g2t.load_state_dict(torch.load(config['g2t']['save']+'X'+'best'))
    model_t2g.load_state_dict(torch.load(config['t2g']['save']+'X'+'best'))
    logging.info('Final Test mode {0:}'.format(config['main']['mode']))
    e = eval_t2g(pool, 'test', vocab, model_t2g, config['t2g'])
    e = eval_g2t(pool, 'test', vocab, model_g2t, config['g2t'])

def multi_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    for i in range(100): 
        config = yaml.load(open(args.config, 'r'))
        config['main']['seed'] = random.randint(0,1e8)
        config['main']['mode'] = random.choice(['sup', 'warm_unsup', 'cold_unsup'])
        config['main']['display'] = False
        save_str = str(random.randint(0,1e5))
        config['g2t']['save'] += save_str
        config['t2g']['save'] += save_str
        _config = copy.deepcopy(config)
        random.seed(config['main']['seed'])
        torch.manual_seed(config['main']['seed'])
        np.random.seed(config['main']['seed'])
        torch.cuda.manual_seed_all(config['main']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        vocab = prep_data(config['main'])
        torch.save({'vocab':vocab}, 'tmp_vocab.pt'+str(config['main']['seed']))
        train('train', config, load='tmp_vocab.pt'+str(config['main']['seed']))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    _config = copy.deepcopy(config)
    random.seed(config['main']['seed'])
    torch.manual_seed(config['main']['seed'])
    np.random.seed(config['main']['seed'])
    torch.cuda.manual_seed_all(config['main']['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    vocab = prep_data(config['main'])
    torch.save({'vocab':vocab}, 'tmp_vocab.pt')
    train('train', config)

if __name__=='__main__':
    main()
    #multi_run()
