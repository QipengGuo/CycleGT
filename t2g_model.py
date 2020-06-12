import torch as tc
import torch.nn as nn
import torch.nn.functional as F

class ModelLSTM(nn.Module):
        def __init__(self , relation_types = 7 , dropout = 0.0, d_model=768):
                super().__init__()

                self.d_model = d_model
                self.relation_types = relation_types
                self.dropout = dropout

                self.emb = nn.Embedding(40000, self.d_model) # 40000 because we use the Bert tokenizer
                self.lstm = nn.LSTM(self.d_model, self.d_model//2, batch_first=True, bidirectional=True, num_layers=2)
                self.wq = nn.Linear(self.d_model , self.d_model)
                self.wk = nn.Linear(self.d_model , self.d_model)
                self.drop = nn.Dropout(self.dropout)
                self.ln1 = nn.Linear(self.d_model , self.d_model)
                self.lno = nn.Linear(self.d_model , relation_types)
                self.blind = False
                self.ln = nn.LayerNorm(self.d_model)

                self.reset_params()

        def reset_params(self):
                nn.init.xavier_normal_(self.wq.weight.data)
                nn.init.xavier_normal_(self.wk.weight.data)
                nn.init.xavier_normal_(self.ln1.weight.data)
                nn.init.xavier_normal_(self.lno.weight.data)

                nn.init.constant_(self.wq.bias.data , 0)
                nn.init.constant_(self.wk.bias.data , 0)
                nn.init.constant_(self.ln1.bias.data , 0)
                nn.init.constant_(self.lno.bias.data , 0)

        def forward(self , batch):
                d = self.d_model

                sents = batch['sents']
                ents = batch['ents']
                if self.blind: #blind means using entity only
                    s = tc.zeros_like(sents)
                else:
                    s = sents
                bs , n = sents.size()
                ne = max([len(x) for x in ents])

                ent_index = s.new_zeros(s.size())
                for _b in range(len(ents)):
                        for u,v in ents[_b]:
                                ent_index[_b , u:v] = 1
                                if self.blind:
                                    s[_b, u:v] = sents[_b, u:v]

                sent_mask  = (s != 0)

                encoded, _ = self.lstm(self.emb(s))
                ent_mask = sent_mask.new_zeros(bs, ne).float()
                ent_encode = encoded.new_zeros(bs, ne, d)
                for _b in range(bs):
                        for i , (u , v) in enumerate(ents[_b]):
                                if u<v:
                                    ent_encode[_b , i] = encoded[_b , u : v , :].mean(dim = 0)
                                ent_mask[_b , i] = 1
                ent_encode = self.ln(ent_encode)

                q = self.wq(ent_encode)
                k = self.wk(ent_encode)
                alpha = q.view(bs,ne,1,d) + k.view(bs,1,ne,d) #(bs , n , n , d)
                alpha = F.relu(self.drop(alpha))
                alpha = F.relu(self.ln1(alpha))
                alpha = self.lno(alpha)


                alpha = alpha * ent_mask.view(bs,ne,1,1) * ent_mask.view(bs,1,ne,1)

                return tc.log_softmax(alpha, -1)

