from .CLIP import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb



class MyModel(nn.Module):
    def __init__(self, 
                 args):

        super(MyModel, self).__init__()
        self.args = args
        self.encoder, self.preprocess = clip.load(self.args.arch_name, device="cpu")
        if self.args.version == 'V1': # image_text
            if 'image' in self.args.activate_branch:
                self.img_adapter = Projector(in_channels=512, out_channels=512)
            if 'text' in self.args.activate_branch:
                self.txt_adapter = Projector(in_channels=512, out_channels=512)
        elif self.args.version == 'V2':
            self.adapter = Adapter(512)
    

    def forward_text(self, texts):
        prompted_texts = [f"a photo of a {c}" for c in texts]
        text_inputs    = torch.cat([clip.tokenize(txt) for txt in prompted_texts]).cuda()
        text_features  = self.encoder.encode_text(text_inputs)
        return text_features
    
    def forward_image(self, images):
        image_features = self.encoder.encode_image(images)
        return image_features
    
    def forward(self, images, txt_features):
        img_feats = self.forward_image(images)
        if self.args.version == 'V1':
            if 'image' in self.args.activate_branch:
                img_feats = self.img_adapter(img_feats)
            if 'text' in self.args.activate_branch:
                txt_feats = self.txt_adapter(txt_features)
            else:
                txt_feats = txt_features
            logits    = F.linear(F.normalize(img_feats, dim=-1, p=2), F.normalize(txt_feats, dim=-1, p=2))
        elif self.args.version == 'V2':
            logits = self.adapter(img_feats, txt_features)

        return logits 


class Projector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Projector, self).__init__()
        self.ln1 = nn.Sequential(
                       nn.Linear(in_channels, 2*in_channels, bias=False),
                       nn.LayerNorm( 2*in_channels),
                       nn.LeakyReLU(0.1)
        )
        self.ln2 = nn.Sequential(
                       nn.Linear(2 * in_channels, 2*in_channels, bias=False),
                       nn.LayerNorm( 2*in_channels),
                       nn.LeakyReLU(0.1)
        )
        self.ln3 = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.ffn = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        identity = x
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = x + self.ffn(identity)
        return x


class Adapter(nn.Module):
    def __init__(self, num_features):
        super(Adapter, self).__init__()
        self.num_features = num_features
        hdim              = self.num_features
        self.slf_attn   = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def forward(self, img_feats, txt_feats):
        # support: [n_w, d]
        # query: [n_q, d]
        bs                 = img_feats.shape[0]
        img_expand         = img_feats.unsqueeze(1)
        txt_expand         = txt_feats.unsqueeze(0).repeat(bs , 1, 1)
        comb               = torch.cat((img_expand, txt_expand), dim=1)
        comb               = self.slf_attn(comb, comb, comb)
        img_upd, txt_upd   = comb[:, 0, :], comb[:, 1:, :]
        logits             = F.cosine_similarity(img_upd.unsqueeze(1), txt_upd, dim=-1) 
        return logits

        
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # modified
        # import pdb
        # pdb.set_trace()
        # value, idx = torch.sort(attn, dim=-1)
        # selected_idx = idx[:, :, :5].squeeze()
        # mask = torch.zeros(*idx.size()).to(idx.device)
        # for j in range(selected_idx.shape[0]):
        #     mask[j, :, selected_idx[j]] = 1
        output = torch.bmm(attn, v)
        return output, attn, log_attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, w_res: bool=True):
        super().__init__()
        self.w_res = w_res
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        if self.w_res:
            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)

        return output 
        # return output, attn # delete later    