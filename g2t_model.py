import dgl
import torch
import math
from data import *
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import torch.utils.data

def replace_ent(x, ent, V, emb):
    # replace the entity
    mask = (x>=V).float()
    _x = emb((x*(1.-mask) + 3 * mask ).long())
    if mask.sum()==0:
        return _x
    idx = ((x-V)*mask + 0 * (1.-mask)).long()
    return _x * (1.-mask[:,None]) + mask[:,None] * ent[torch.arange(len(idx)).cuda(), idx].view(_x.shape)

def len2mask(lens, device):
    max_len = max(lens)
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lens), max_len)
    mask = mask >= torch.LongTensor(lens).to(mask).unsqueeze(1)
    return mask

class MSA(nn.Module):
    # Multi-head Self Attention
    def __init__(self, args, mode='normal'):
        super(MSA, self).__init__()
        if mode=='copy':
            nhead, head_dim = 1, args.nhid
            qninp, kninp = args.dec_ninp, args.nhid
        if mode=='normal':
            nhead, head_dim = args.nhead, args.head_dim
            qninp, kninp = args.nhid, args.nhid
        self.attn_drop = nn.Dropout(0.1)
        self.WQ = nn.Linear(qninp, nhead*head_dim, bias=True if mode=='copy' else False)
        if mode!='copy':
            self.WK = nn.Linear(kninp, nhead*head_dim, bias=False)
            self.WV = nn.Linear(kninp, nhead*head_dim, bias=False)
        self.args, self.nhead, self.head_dim, self.mode = args, nhead, head_dim, mode

    def forward(self, inp1, inp2, mask=None):
        B, L2, H = inp2.shape
        NH, HD = self.nhead, self.head_dim
        if self.mode=='copy':
            q, k, v = self.WQ(inp1), inp2, inp2
        else:
            q, k, v = self.WQ(inp1), self.WK(inp2), self.WV(inp2)
        L1 = 1 if inp1.ndim==2 else inp1.shape[1]
        if self.mode!='copy':
            q = q / math.sqrt(H)
        q = q.view(B, L1, NH, HD).permute(0, 2, 1, 3) 
        k = k.view(B, L2, NH, HD).permute(0, 2, 3, 1)
        v = v.view(B, L2, NH, HD).permute(0, 2, 1, 3)
        pre_attn = torch.matmul(q,k)
        if mask is not None:
            pre_attn = pre_attn.masked_fill(mask[:,None,None,:], -1e8)
        if self.mode=='copy':
            return pre_attn.squeeze(1)
        else:
            alpha = self.attn_drop(torch.softmax(pre_attn, -1))
            attn = torch.matmul(alpha, v).permute(0, 2, 1, 3).contiguous().view(B,L1,NH*HD)
            ret = attn
            if inp1.ndim==2:
                return ret.squeeze(1)
            else:
                return ret

class BiLSTM(nn.Module):
    # for entity encoding
    def __init__(self, args, enc_type='title'):
        super(BiLSTM, self).__init__()
        self.enc_type = enc_type
        self.drop = nn.Dropout(args.emb_drop)
        self.bilstm = nn.LSTM(args.nhid, args.nhid//2, bidirectional=True, \
                num_layers=args.enc_lstm_layers, batch_first=True)
 
    def forward(self, inp, mask, ent_len=None):
        inp = self.drop(inp)
        lens = (mask==0).sum(-1).long().tolist()
        pad_seq = pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted=False)
        y, (_h, _c) = self.bilstm(pad_seq)
        if self.enc_type=='entity':
            _h = _h.transpose(0,1).contiguous()
            _h = _h[:,-2:].view(_h.size(0), -1) # two directions of the top-layer
            ret = pad(_h.split(ent_len), out_type='tensor')
            return ret

class GAT(nn.Module):
    # a graph attention network with dot-product attention
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 ffn_drop=0.,
                 attn_drop=0.,
                 trans=True):
        super(GAT, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.q_proj = nn.Linear(in_feats, num_heads*out_feats, bias=False)
        self.k_proj = nn.Linear(in_feats, num_heads*out_feats, bias=False)
        self.v_proj = nn.Linear(in_feats, num_heads*out_feats, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm(in_feats)
        self.ln2 = nn.LayerNorm(in_feats)
        if trans:
            self.FFN = nn.Sequential(
                nn.Linear(in_feats, 4*in_feats),
                nn.PReLU(4*in_feats),
                nn.Linear(4*in_feats, in_feats),
                nn.Dropout(0.1),
            )
            # a strange FFN
        self._trans = trans

    def forward(self, graph, feat):
        graph = graph.local_var()
        feat_c = feat.clone().detach().requires_grad_(False)
        q, k, v = self.q_proj(feat), self.k_proj(feat_c), self.v_proj(feat_c)
        q = q.view(-1, self._num_heads, self._out_feats)
        k = k.view(-1, self._num_heads, self._out_feats)
        v = v.view(-1, self._num_heads, self._out_feats)
        graph.ndata.update({'ft': v, 'el': k, 'er': q}) # k,q instead of q,k, the edge_softmax is applied on incoming edges
        # compute edge attention
        graph.apply_edges(fn_u_dot_v('el', 'er', 'e'))
        e =  graph.edata.pop('e') / math.sqrt(self._out_feats * self._num_heads)
        graph.edata['a'] = edge_softmax(graph, e).unsqueeze(-1)
       # message passing
        graph.update_all(fn_u_mul_e('ft', 'a', 'm'),
                         fn_sum('m', 'ft2'))
        rst = graph.ndata['ft2']
        # residual
        rst = rst.view(feat.shape) + feat
        if self._trans:
            rst = self.ln1(rst)
            rst = self.ln1(rst+self.FFN(rst))
            # use the same layer norm
        return rst

class GraphTrans(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        if args.graph_enc == "gat":
            # we only support gtrans, don't use this one
            self.gat = nn.ModuleList([GAT(args.nhid, args.nhid//4, 4, attn_drop=args.attn_drop, trans=False) for _ in range(args.prop)]) #untested
        else:
            self.gat = nn.ModuleList([GAT(args.nhid, args.nhid//4, 4, attn_drop=args.attn_drop, ffn_drop=args.drop, trans=True) for _ in range(args.prop)])
        self.prop = args.prop

    def forward(self, ent, ent_mask, ent_len, rel, rel_mask, graphs):
        device = ent.device
        ent_mask = (ent_mask==0) # reverse mask
        rel_mask = (rel_mask==0)
        init_h = []
        for i in range(graphs.batch_size):
            init_h.append(ent[i][ent_mask[i]])
            init_h.append(rel[i][rel_mask[i]])
        init_h = torch.cat(init_h, 0)
        feats = init_h
        if graphs.number_of_nodes()!=len(init_h):
            print('Err', graphs.number_of_nodes(), len(init_h), ent_mask, rel_mask)
        else:
            for i in range(self.prop):
                feats = self.gat[i](graphs, feats)
        g_root = feats.index_select(0, graphs.filter_nodes(lambda x: x.data['type']==NODE_TYPE['root']).to(device))
        g_ent = pad(feats.index_select(0, graphs.filter_nodes(lambda x: x.data['type']==NODE_TYPE['entity']).to(device)).split(ent_len), out_type='tensor')
        return g_ent, g_root


def fn_u_dot_v(n1,n2,n3):
    def func(edge_batch):
        return {n3: torch.matmul(edge_batch.src[n1].unsqueeze(-2), edge_batch.dst[n2].unsqueeze(-1)).squeeze(-1).squeeze(-1)}
    return func

def fn_u_mul_e(n1,n2,n3):
    def func(edge_batch):
        return {n3: edge_batch.src[n1] * edge_batch.data[n2]}
    return func

def fn_sum(n1, n2):
    def func(node_batch):
        return {n2: node_batch.mailbox[n1].sum(1)}
    return func          
class GraphWriter(nn.Module):
    def __init__(self, args, vocab_pack=None):
        super(GraphWriter, self).__init__()
        if vocab_pack is not None:
            args.rel_vocab = vocab_pack['relation']
            args.text_vocab = vocab_pack['text']
            args.ent_text_vocab = vocab_pack['entity']

        args.dec_ninp = args.nhid * 2
        self.text_vocab_len = len(args.text_vocab) # to fix this length
        self.args = args
        self.ent_emb = nn.Embedding(len(args.ent_text_vocab), args.nhid, padding_idx=0)
        self.tar_emb = nn.Embedding(self.text_vocab_len, args.nhid, padding_idx=0)
        nn.init.xavier_normal_(self.ent_emb.weight)
        self.rel_emb = nn.Embedding(len(args.rel_vocab), args.nhid, padding_idx=0)
        nn.init.xavier_normal_(self.rel_emb.weight)
        self.decode_lstm = nn.LSTMCell(args.dec_ninp, args.nhid)
        self.ent_enc = BiLSTM(args, enc_type='entity')
        self.graph_enc = GraphTrans(args)
        self.ent_attn = MSA(args)
        self.copy_attn = MSA(args, mode='copy')
        self.blind = False
        self.copy_fc = nn.Linear(args.dec_ninp, 1)
        self.pred_v_fc = nn.Linear(args.dec_ninp, self.text_vocab_len)
        self.ln = nn.LayerNorm(args.nhid)

    def enc_forward(self, batch, ent_mask, ent_text_mask, ent_len, rel_mask):
        ent_enc = self.ent_enc(self.ent_emb(batch['ent_text']), ent_text_mask, ent_len = batch['ent_len'])
        rel_emb = self.rel_emb(batch['rel'])
        if self.blind:
            g_ent, g_root = ent_enc, ent_enc.mean(1)
        else:
            g_ent, g_root = self.graph_enc(ent_enc, ent_mask, ent_len, rel_emb, rel_mask, batch['graph'])
        return self.ln(g_ent), g_root, ent_enc

    def forward(self, batch, beam_size=-1):
        # three modes, beam_size==-1 means training, beam_size==1 means greedy decoding, beam_size>1 means beam search
        ent_mask = len2mask(batch['ent_len'], batch['ent_text'].device)
        ent_text_mask = batch['ent_text']==0

        rel_mask = batch['rel']==0 # 0 means the <PAD>
        g_ent, g_root, ent_enc = self.enc_forward(batch, ent_mask, ent_text_mask, batch['ent_len'], rel_mask)

        _h, _c = g_root, g_root.clone().detach()
        ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
        if beam_size<1:
            # training
            outs = []
            _mask = (batch['text']>=len(self.args.text_vocab)).long()
            _inp = _mask * 3 + (1.-_mask) * batch['text'] # 3 is <UNK> 
            tar_inp = self.tar_emb(_inp.long())
            tar_inp = (1.-_mask[:,:,None]) * tar_inp +  ent_enc[torch.arange(len(batch['text']))[:,None].cuda(),((batch['text']-len(self.args.text_vocab)) * _mask ).long()] * _mask[:,:,None]

            tar_inp = tar_inp.transpose(0,1)
            for t, xt in enumerate(tar_inp):
                _xt = torch.cat([ctx, xt], 1)
                _h, _c = self.decode_lstm(_xt, (_h, _c))
                ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                outs.append(torch.cat([_h, ctx], 1))
            outs = torch.stack(outs, 1)
            copy_gate = torch.sigmoid(self.copy_fc(outs))
            EPSI = 1e-6
            # copy
            pred_v = torch.log(copy_gate+EPSI) + torch.log_softmax(self.pred_v_fc(outs), -1)
            pred_c = torch.log((1. - copy_gate)+EPSI) + torch.log_softmax(self.copy_attn(outs, ent_enc, mask=ent_mask), -1)
            pred = torch.cat([pred_v, pred_c], -1)
            return pred, torch.exp(pred_c)
        else:
            if beam_size==1:
                # greedy
                device = g_ent.device
                B = g_ent.shape[0]
                seq = (torch.ones(B,).long().to(device) * self.args.text_vocab('<BOS>')).unsqueeze(1)
                for t in range(self.args.beam_max_len):
                    _inp = seq[:,-1]
                    xt = replace_ent(seq[:,-1], ent_enc, len(self.args.text_vocab), self.tar_emb)
                    _xt = torch.cat([ctx, xt], 1)
                    _h, _c = self.decode_lstm(_xt, (_h, _c))
                    ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                    _y = torch.cat([_h, ctx], 1)
                    copy_gate = torch.sigmoid(self.copy_fc(_y))
                    pred_v = torch.log(copy_gate) + torch.log_softmax(self.pred_v_fc(_y), -1)
                    pred_c = torch.log((1. - copy_gate)) + torch.log_softmax(self.copy_attn(_y.unsqueeze(1), ent_enc, mask=ent_mask).squeeze(1), -1)
                    pred = torch.cat([pred_v, pred_c], -1).view(B,-1)
                    for ban_item in ['<BOS>', '<PAD>', '<UNK>']:
                        pred[:, self.args.text_vocab(ban_item)] = -1e8
                    _, word = pred.max(-1)
                    seq = torch.cat([seq, word.unsqueeze(1)], 1)
                    eos_idx = self.args.text_vocab('<EOS>')
                    if ((seq==eos_idx).float().max(-1)[0]==1).all():
                        break
                return seq
            else:
                # beam search
                device = g_ent.device
                B = g_ent.shape[0]
                BSZ = B * beam_size
                _h = _h.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                _c = _c.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                ent_mask = ent_mask.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                ctx = ctx.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                g_ent = g_ent.view(B, 1, g_ent.size(1), -1).repeat(1, beam_size, 1, 1).view(BSZ, g_ent.size(1), -1)
                ent_enc = ent_enc.view(B, 1, ent_enc.size(1), -1).repeat(1, beam_size, 1, 1).view(BSZ, ent_enc.size(1), -1)

                beam_best = torch.zeros(B).to(device) - 1e9
                beam_seq = (torch.ones(B, beam_size).long().to(device) * self.args.text_vocab('<BOS>')).unsqueeze(-1)
                beam_best_seq = torch.zeros(B,1).long().to(device)
                beam_score = torch.zeros(B, beam_size).to(device)
                done_flag = torch.zeros(B, beam_size).to(device)
                for t in range(self.args.beam_max_len):
                    _inp = beam_seq[:,:,-1].view(-1)
                    _mask = (_inp>=len(self.args.text_vocab)).long()
                    xt = replace_ent(beam_seq[:,:,-1].view(-1), ent_enc, len(self.args.text_vocab), self.tar_emb)
                    _xt = torch.cat([ctx, xt], 1)
                    _h, _c = self.decode_lstm(_xt, (_h, _c))
                    ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                    _y = torch.cat([_h, ctx], 1)
                    copy_gate = torch.sigmoid(self.copy_fc(_y))
                    pred_v = torch.log(copy_gate) + torch.log_softmax(self.pred_v_fc(_y), -1)
                    pred_c = torch.log((1. - copy_gate)) + torch.log_softmax(self.copy_attn(_y.unsqueeze(1), ent_enc, mask=ent_mask).squeeze(1), -1)
                    pred = torch.cat([pred_v, pred_c], -1).view(B, beam_size, -1)
                    for ban_item in ['<BOS>', '<PAD>', '<UNK>']:
                        pred[:, :, self.args.text_vocab(ban_item)] = -1e8
                    if t==self.args.beam_max_len-1: # force ending 
                        tt = pred[:, :, self.args.text_vocab('<EOS>')]
                        pred = pred*0-1e8
                        pred[:, :, self.args.text_vocab('<EOS>')] = tt
                    cum_score = beam_score.view(B,beam_size,1) + pred
                    score, word = cum_score.topk(dim=-1, k=beam_size) # B, beam_size, beam_size
                    score, word = score.view(B,-1), word.view(B,-1)
                    eos_idx = self.args.text_vocab('<EOS>')
                    if beam_seq.size(2)==1:
                        new_idx = torch.arange(beam_size).to(device)
                        new_idx = new_idx[None,:].repeat(B,1)
                    else:
                        _, new_idx = score.topk(dim=-1, k=beam_size)
                    new_src, new_score, new_word, new_done = [], [], [], []
                    LP = beam_seq.size(2) ** self.args.lp
                    prefix_idx = torch.arange(B).to(device)[:,None]
                    new_word = word[prefix_idx, new_idx]
                    new_score = score[prefix_idx, new_idx]
                    _mask = (new_word==eos_idx).float()
                    _best = _mask * (done_flag==0).float() * new_score
                    _best = _best * (_best!=0) -1e8 * (_best==0)
                    new_src = new_idx//beam_size
                    _best, _best_idx = _best.max(1)
                    _best = _best/LP
                    _best_mask = (_best>beam_best).float()
                    beam_best = beam_best * (1.-_best_mask) + _best_mask * _best
                    beam_best_seq = beam_best_seq * (1.-_best_mask[:,None]) + _best_mask[:,None] * beam_seq[prefix_idx, new_src[prefix_idx, _best_idx[:,None]]].squeeze(1)
                    new_score = -1e8 * _mask + (1.-_mask) * new_score
                    new_done = 1 * _mask + (1.-_mask) * done_flag
                    beam_score = new_score
                    done_flag = new_done
                    beam_seq = beam_seq.view(B,beam_size,-1)[torch.arange(B)[:,None].to(device), new_src]
                    beam_seq = torch.cat([beam_seq, new_word.unsqueeze(2)], 2)
                    beam_best_seq = torch.cat([beam_best_seq, torch.zeros(B,1).to(device)], 1)
                    _h = _h.view(B,beam_size,-1)[torch.arange(B)[:,None].to(device), new_src].view(BSZ,-1)
                    _c = _c.view(B,beam_size,-1)[torch.arange(B)[:,None].to(device), new_src].view(BSZ,-1)
                    ctx = ctx.view(B,beam_size,-1)[torch.arange(B)[:,None].to(device), new_src].view(BSZ,-1)
                    if (done_flag==1).all():
                        break

                return beam_best_seq.long()
                  
            
            
            
            
