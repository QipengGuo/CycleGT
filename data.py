import torch
import dgl
import uuid
import copy
import random
from transformers import BertTokenizer
bert_type = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_type)

NODE_TYPE = {'entity': 0, 'root': 1, 'relation':2}
def pad(var_len_list, out_type='list', flatten=False):
    #padding sequences
    if flatten:
        lens = [len(x) for x in var_len_list]
        var_len_list = sum(var_len_list, [])
    max_len = max([len(x) for x in var_len_list])
    if out_type=='list':
        if flatten:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list], lens
        else:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list]
    if out_type=='tensor':
        if flatten:
            return torch.stack([torch.cat([x, \
            torch.zeros([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0), lens
        else:
            return torch.stack([torch.cat([x, \
            torch.zeros([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0)

def write_txt(batch, seqs, text_vocab):
    # converting the prediction to real text.
    ret = []
    for b, seq in enumerate(seqs):
        txt = []

        for token in seq:
            # copy the entity
            if token>=len(text_vocab):
                if (token-len(text_vocab))>=len(batch['raw_ent_text'][b]):
                    print((token-len(text_vocab)), len(batch['raw_ent_text'][b]))
                    tok = ['NO_ENT']
                else:
                    tok = batch['raw_ent_text'][b][token-len(text_vocab)]
                    #tok = ['ENT_'+str(int(token-len(text_vocab)))+'_ENT'] 
                ent_text = tok 
                ent_text = filter(lambda x:x!='<PAD>', ent_text)
                txt.extend(ent_text)
            else:
                if int(token) not in [text_vocab(x) for x in ['<PAD>', '<BOS>', '<EOS>']]:
                    txt.append(text_vocab(int(token)))
            if int(token) == text_vocab('<EOS>'):
                break
        ret.append([' '.join([str(x) for x in txt]).replace('<BOS>', '').replace('<EOS>', '')])
    return ret

class Vocab(object):
    def __init__(self, max_vocab=2**31, min_freq=-1, sp=['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<ROOT>']):
        self.i2s = []
        self.s2i = {}
        self.wf = {}
        self.inv = {}
        self.max_vocab, self.min_freq, self.sp = max_vocab, min_freq, copy.deepcopy(sp)

    def __len__(self):
        return len(self.i2s)

    def __str__(self):
        return 'Total ' + str(len(self.i2s)) + str(self.i2s[:10])

    def merge(self, _vocab):
        self.wf.update(_vocab.wf)
        self.inv.update(_vocab.inv)
        self.sp = list(set(self.sp + _vocab.sp))

    def update(self, token, inv=False, sp=False):
        if isinstance(token, list):
            for t in token:
                self.update(t, inv=inv, sp=sp)
        else:
            self.wf[token] = self.wf.get(token, 0) + 1
            if inv:
                self.wf[token+'_INV'] = self.wf.get(token+'_INV', 0) + 1
                self.inv[token] = token+'_INV'
            if sp and token not in self.sp:
                self.sp.append(token)

    def get_inv(self, idx):
        return self.__call__(self.inv.get(self.i2s[idx], '<UNK>'))

    def build(self):
        self.i2s.extend(self.sp)
        sort_kv = sorted(self.wf.items(), key=lambda x:x[1], reverse=True)
        for k,v in sort_kv:
            if len(self.i2s)<self.max_vocab and v>=self.min_freq and k not in self.sp:
                self.i2s.append(k)
        self.s2i.update(list(zip(self.i2s, range(len(self.i2s)))))

    def __call__(self, x, ents=[]):
        if isinstance(x, list):
            return [self(y) for y in x]
        if isinstance(x, int):
            if x>=len(self.i2s):
                return ents[int(x-len(self.i2s))]
            return self.i2s[x]
        else:
            if x[0] == '<' and x[-1] == '>' and '_' in x:
                try:
                    t = len(self.s2i)+int(x.split('_')[1][:-1])
                except:
                    print(x)
                return len(self.s2i)+int(x.split('_')[1][:-1])
            return self.s2i.get(x, self.s2i['<UNK>'])

def scan_data(datas, vocab=None, sp=False):
    MF = -1

    if vocab is None:
        vocab = {'text':Vocab(min_freq=MF), 'entity':Vocab(min_freq=MF), 'relation':Vocab()}
    for data in datas:
        vocab['text'].update(data['text'].split(), sp=sp)
        vocab['entity'].update(sum(data['entities'], []), sp=sp)
        vocab['relation'].update([x[1] for x in data['relations']], inv=True)
    return vocab

def get_graph(ent_len, rel_len, adj_edges):
    graph = dgl.DGLGraph()

    graph.add_nodes(ent_len,
                    {'type': torch.ones(ent_len) * NODE_TYPE['entity']})
    graph.add_nodes(1, {'type': torch.ones(1) * NODE_TYPE['root']})
    graph.add_nodes(rel_len * 2,
                    {'type': torch.ones(rel_len * 2) * NODE_TYPE['relation']})
    graph.add_edges(ent_len, torch.arange(ent_len))
    graph.add_edges(torch.arange(ent_len), ent_len)
    graph.add_edges(torch.arange(ent_len + 1 + rel_len * 2),
                    torch.arange(ent_len + 1 + rel_len * 2))

    if len(adj_edges) > 0:
        graph.add_edges(*list(map(list, zip(*adj_edges))))
    return graph

def build_graph(ent_len, relations):
    rel_len = len(relations)

    adj_edges = []
    for i, r in enumerate(relations):
        st_ent, rt, ed_ent = r
        # according to the edge_softmax operator, we need to reverse the graph
        adj_edges.append([ent_len+1+2*i, st_ent])
        adj_edges.append([ed_ent, ent_len+1+2*i])
        adj_edges.append([ent_len+1+2*i+1, ed_ent])
        adj_edges.append([st_ent, ent_len+1+2*i+1])

    graph = get_graph(ent_len, rel_len, adj_edges)
    return graph

class Example(object):
    def __init__(self, data, vocab):
        self.uuid = uuid.uuid4()
        self.vocab = vocab
        self.text = [vocab['text'](x) for x in data['text'].split()]
        self.entities = [vocab['entity'](x) for x in data['entities']]
        self.relations = []
        for r in data['relations']:
            e1, e2 = vocab['entity'](r[0]), vocab['entity'](r[2])
            rel = vocab['relation'](r[1])
            e1, e2 = self.entities.index(e1), self.entities.index(e2)
            self.relations.append([e1, rel, e2])

        self.graph = None
        self.graph = build_graph(len(self.entities), self.relations)
        self.id = None

    def __str__(self):
        return '\n'.join(
            [str(k) + ':\t' + str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.text)


    def get(self):
        if hasattr(self, '_cached_tensor') and False:
            return self._cached_tensor
        else:
            vocab = self.vocab
            ret = {}
            ret['text'] = [vocab['text']('<BOS>')] + self.text + [vocab['text']('<EOS>')]
            ret['ent_text'] = [[vocab['entity']('<BOS>')] + x + [vocab['entity']('<EOS>')] for x in self.entities]
            ret['relation'] = [vocab['relation']('<ROOT>')] + sum([[x[1], vocab['relation'].get_inv(x[1])] for x in self.relations], [])
            ret['raw_relation'] = self.relations
            ret['graph'] = self.graph
            ret['uuid'] = self.uuid

            self._cached_tensor = ret
            return self._cached_tensor

class DataPool(object):
    def __init__(self):
        self.pool = []
        self.types={}

    def add(self, data, _type='gold'):
        self.pool.append(data)
        if _type not in self.types:
            self.types[_type] = [] 
        self.types[_type].append(len(self.pool)-1)

    def remove(self, _type, _id):
        del self.types[_type][_id]

    def __len__(self):
        return len(self.pool)

    def get_len(self, _type):
        return len(self.types.get(_type, []))

    def draw_with_type(self, batch_size=32, shuffle=True, _type='gold', tot=1.1):
        batch = []
        from copy import deepcopy
        if shuffle:
            random.shuffle(self.types[_type])
        for i,idx in enumerate(self.types[_type]):
            batch.append(deepcopy(self.pool[idx]))
            if len(batch)>=batch_size:
                yield batch
                batch = []
            if i>tot*len(self.types[_type]):
                break
        if len(batch)>0:
            yield batch

def batch2tensor_g2t(datas, device, vocab):
    # raw batch to tensor 
    ret = {}
    ret['ent_len'] = [len(x['ent_text']) for x in datas]
    ents = [vocab['entity'](x['ent_text']) for x in datas]
    ret['raw_ent_text'] = ents
    ret['text'] = pad([torch.LongTensor(x['text']) for x in datas], 'tensor').to(device)
    ret['tgt'] = ret['text'][:,1:]
    ret['text'] = ret['text'][:,:-1]
    ent_text = sum([[torch.LongTensor(y) for y in x['ent_text']] for x in datas], [])
    ret['ent_text'] = pad(ent_text, 'tensor').to(device)
    ret['rel'] = pad([torch.LongTensor(x['relation']) for x in datas], 'tensor').to(device)
    ret['graph'] = dgl.batch([x['graph'] for x in datas]).to(device)
    return ret


def tensor2data_g2t(old_data, pred):
    # construct synthetic data based on the old data and model prediction
    new_data = {}
    new_data['text'] = pred
    new_data['ent_text'] = old_data['ent_text']
    new_data['relation'] = old_data['relation']
    new_data['raw_relation'] = old_data['raw_relation']
    new_data['graph'] = old_data['graph']
    new_data['uuid'] = old_data['uuid']
    return new_data

def tensor2data_t2g(old_data, pred, vocab):
    # construct synthetic data based on the old data and model prediction
    new_data = {}
    new_data['text'] = old_data['text']
    new_data['ent_text'] = old_data['ent_text']
    new_data['relation'] = [vocab['relation']('<ROOT>')] + sum([[x[1], vocab['relation'].get_inv(x[1])] for x in pred], [])
    new_data['graph'] = build_graph(len(new_data['ent_text']), pred)
    new_data['uuid'] = old_data['uuid']
    return new_data

def batch2tensor_t2g(datas, device, vocab, add_inp=False):
    # raw batch to tensor, we use the Bert tokenizer for the T2G model
    ret = {}
    ent_pos = []
    text = []
    tgt = []
    MAX_ENT = 100
    ent_len = 1
    for data in datas:
        ents = [vocab['entity'](x) for x in data['ent_text']]
        st, ed = [], []
        cstr = ''
        ent_order = []
        for i, t in enumerate(data['text']):
            if t>=len(vocab['text']):
                ff = (t-len(vocab['text'])) not in ent_order
                if ff:
                    st.append(len(cstr))
                cstr += ' '.join([x for x in vocab['text'](t, ents) if x[0]!='<' and x[-1]!='>'])
                if ff:
                    ent_order.append(t-len(vocab['text']))
                    ed.append(len(cstr))
            else:
                if vocab['text'](t)[0]=='<':
                    continue
                cstr += vocab['text'](t)
            cstr += '' if i==len(data['text'])-1 else ' '
        if add_inp:
            cstr += ' ' + ' '.join([' '.join(e) for e in ents])
        tok_abs = ["[CLS]"] + tokenizer.tokenize(cstr) + ["[SEP]"]
        _ent_pos = []
        for s,e in zip(st, ed):
            guess_start = s - cstr[:s].count(" ") + 5
            guess_end   = e - cstr[:e].count(" ") + 5 

            new_s = -1
            new_e = -1
            l = 0
            r = 0
            for i in range(len(tok_abs)):
                l = r
                r = l + len(tok_abs[i]) - tok_abs[i].count("##")*2
                if l <= guess_start and guess_start < r:
                    new_s = i
                if l <= guess_end and guess_end < r:
                    new_e = i
            _ent_pos.append((new_s, new_e))
        _order_ent_pos = []
        for _e in range(len(ents)):
            if _e in ent_order:
                idx = ent_order.index(_e)
                _order_ent_pos.append(_ent_pos[idx])
            else:
                idx = 0
                _order_ent_pos.append((0, 1))

        ent_pos.append(_order_ent_pos)
        text.append(tokenizer.convert_tokens_to_ids(tok_abs))
        _tgt = torch.zeros(MAX_ENT, MAX_ENT)
        _tgt[:len(_ent_pos), :len(_ent_pos)] += 3 # <UNK>
        for _e1, _r, _e2 in data['raw_relation']:
            if _e1 not in ent_order or _e2 not in ent_order: # the synthetic data may lose some entities
                continue
            _tgt[_e1, _e2] = _r
        tgt.append(_tgt)
        ent_len = max(ent_len, len(_order_ent_pos))
    ret['sents'] = pad([torch.LongTensor(x) for x in text], 'tensor').to(device)
    ret['ents'] = ent_pos
    ret['tgt'] = torch.stack(tgt,0)[:,:ent_len,:ent_len].long().to(device)
    return ret

def fill_pool(pool, vocab, datas, _type):
    for data in datas:
        ex = Example(data, vocab).get()
        pool.add(ex, _type)

