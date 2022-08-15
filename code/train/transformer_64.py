import os
import os.path as osp
import copy
import math
from typing import List, Optional
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torchvision.transforms as T

from backbone import Joiner, build_backbone
from patch_embedding import Embeddings
from position_encoding import PositionEmbeddingSine
from util.msic import nested_tensor_from_tensor_list
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image


class Generator(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=256*4, dropout=0.1,
                 activation="relu"):
        super(Generator, self).__init__()
        

        encoder_layer = TransformerEncoderLayer(256, nhead, 256*4, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.embeddings = Embeddings()
        
        decoder_layer = TransformerDecoderLayer(384, 1, 384*4, dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.value1 = nn.Linear(384,1536)
        self.deconv = nn.Sequential(
            nn.Conv2d(384//16, 3, 1, 1, 0)
        )
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)
        
        self.aq = nn.Linear(384, 384)
        self.ak = nn.Linear(384, 384)
        self.anorm = nn.LayerNorm(384)
        self.adropout = nn.Dropout(0.1)
        
        self.avg = nn.AvgPool2d((4,4),(2,2),1,count_include_pad=False)
        self._reset_parameters()
        self.backbone = build_backbone()
        self.num_decoder_layers = num_decoder_layers
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask):
        image = src
        maski = mask
        rotia = 0.25
        src1 = F.interpolate(src, scale_factor = rotia)
        mask1 = F.interpolate(mask, scale_factor = rotia)
        B,C,H,W = src.size()
        embedding, in_mask, in_mask1, in_mask2 = self.embeddings(src1,mask1)
        xx = torch.zeros_like(src1).cuda()
        xm = torch.zeros_like(src1).cuda()
        if isinstance(src, (list, torch.Tensor)):
            src = nested_tensor_from_tensor_list(src * mask)
        features, pos = self.backbone(src)
        src, _ = features[-1].decompose()

        ##Transformer Texture Encoder
        memory = self.encoder(self.input_proj(src).flatten(2).permute(2, 0, 1), pos[-1].flatten(2).permute(2, 0, 1)).permute(1,0,2)
        
        ## Coarse Filled Attention
        embedding1 = embedding
        for u in range(0,64):
            e2 = self.aq(embedding[torch.where(in_mask2[:]==64-(u+1))[0],torch.where(in_mask2[:]==64-(u+1))[1]].reshape(embedding.shape[0],-1,embedding.shape[-1])).view(embedding.shape[0],-1,embedding.shape[-1])
            eu = embedding[torch.where(in_mask2[:]>=64-(u))[0],torch.where(in_mask2[:]>=64-(u))[1]].reshape(embedding.shape[0],-1,embedding.shape[-1])
            e1 = self.ak(eu).view(embedding.shape[0],-1,embedding.shape[-1])
            if e2.shape[1] == 0:
                continue
            att = (e2 @ e1.transpose(-2, -1)) * (1.0 / math.sqrt(e1.size(-1)))
            att = F.softmax(att, dim=-1)
            e2 = att @ eu
            e2 = e2.view(embedding.shape[0],e2.shape[1],embedding.shape[-1])
            
            tmp = np.linspace(0,B-1,B,dtype=np.int)
            
            for i in range(e2.shape[1]):
                embedding[tmp,torch.where(in_mask2[:]==64-(u+1))[1][i], :] = e2[tmp,i,:]
                
        embedding = embedding1 + self.adropout(embedding)
        embedding = self.anorm(embedding)
        
        lamed = [0.1,0.1,0.1,0.1,0.1,0.1]
        order = torch.cumsum(in_mask,axis=1)
        
        ## Transformer Structure Decoder
        out_c, score_c = self.decoder(embedding[torch.where(in_mask1[:]==16)[0],torch.where(in_mask1[:]==16)[1]].reshape(embedding.shape[0],-1,embedding.shape[-1]),
                                         memory)    
        if int(order[0,63].data) + 1 > 49:
            N_patch = 65
        else:
            N_patch = 17 +  (int(order[0,63].data))
        
        for u in range(int(order[0,63].data) + 1,N_patch): 
            u_out, u_score = self.decoder(embedding[torch.where(in_mask[:]==0)[0],torch.where(in_mask[:]==0)[1]].reshape(embedding.shape[0],-1,embedding.shape[-1]),memory,lamed,score_c,out_c)
            t = torch.argmax(torch.sum((u_score[self.num_decoder_layers-1]),dim=3).squeeze(1),dim = 1)
            ods = order[torch.where(in_mask[:]==0)[0],torch.where(in_mask[:]==0)[1]].squeeze(0).reshape(order.shape[0],-1)
            tmp = np.linspace(0,B-1,B,dtype=np.int)##batch_size
            index = (t.data + ods[tmp,t].data).cpu().numpy()
            out_rgb = (u_out[self.num_decoder_layers])
            
            x = self.deconv((self.value1(out_rgb[tmp,t,:]).view(-1, 24,8, 8)))
            
            h = (((index)  // 8) * 8).astype(np.int)
            w = (((index) % 8) * 8).astype(np.int)
            
            xxs = torch.zeros_like(xx[:,:,0:8,0:8]).cuda()
            for i in range(xx.shape[0]):
                xxs[i,:,:,:] = x[i,:,:,:]  * (1-mask1[i,:,h[i]:h[i]+8,w[i]:w[i]+8])+src1[i,:,h[i]:h[i]+8,w[i]:w[i]+8]* mask1[i,:,h[i]:h[i]+8,w[i]:w[i]+8]
                xx[i,:,h[i]:h[i]+8,w[i]:w[i]+8] = xxs[i,:,:,: ]
                xm[i,:,h[i]:h[i]+8,w[i]:w[i]+8] = 1
            f = self.embeddings(xxs,mask1,0)
            f = f.flatten(2)
            f = f.transpose(-1, -2)
                
            out_c, score_c = self.decoder(f, memory, [2,2,2,2,2,2],score_c,out_c)
            in_mask[tmp,index] = 1
            order = torch.cumsum(in_mask,axis=1)
            
        if N_patch != 65 :
            u_out, u_score = self.decoder(embedding[torch.where(in_mask[:]==0)[0],torch.where(in_mask[:]==0)[1]].reshape(embedding.shape[0],-1,embedding.shape[-1]),memory,lamed,score_c,out_c)
            u_sort = torch.argsort(torch.sum((u_score[self.num_decoder_layers-1]),dim=3).squeeze(1),dim = 1).T
        for u in range(N_patch,65):
            t = u_sort[u - N_patch,:]
            tmp = np.linspace(0,B-1,B,dtype=np.int)##batch_size
            index = (torch.where(in_mask[:]==0)[1].reshape(xx.shape[0],-1)[tmp,t]).cpu().numpy()
            out_rgb = (u_out[self.num_decoder_layers])
            
            x = self.deconv((self.value1(out_rgb[tmp,t,:]).view(-1, 24,8, 8)))
            h = (((index) // 8) * 8).astype(np.int)
            w = (((index) % 8) * 8).astype(np.int)
            for i in range(xx.shape[0]):
                xx[i,:,h[i]:h[i]+8,w[i]:w[i]+8] = x[i,:,:,:] * (1-mask1[i,:,h[i]:h[i]+8,w[i]:w[i]+8])+src1[i,:,h[i]:h[i]+8,w[i]:w[i]+8]* mask1[i,:,h[i]:h[i]+8,w[i]:w[i]+8]
                xm[i,:,h[i]:h[i]+8,w[i]:w[i]+8] = 1
            
            
        xt = self.avg(src1*mask1+xx*(1-mask1))
        xt = F.interpolate(xt, scale_factor = 2,mode='bilinear')
        xx = (xt * (1-xm) + xx * xm)*(1-mask1)+src1*mask1
        
        return xx
 

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)
        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256*4, dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Generator(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    def forward(self, tgt, memory, a1=[-1,-1,-1,-1,-1,-1],score_c=[None,None,None,None,None,None],out_c=[None,None,None,None,None,None]):#
        output = tgt
        score2 = []
        outputs=[]
        for num, layer in enumerate(self.layers):
            if a1[num]==2:
                outputs.append(torch.cat((out_c[num],output),-2))
                output,score2s = layer(output,memory,a1[num], score_c[num],torch.cat((out_c[num],output),-2))
                score2.append(torch.cat((score_c[num],score2s),-2))
            else: 
                outputs.append(output)
                output,score2s = layer(output,memory,a1[num], score_c[num],out_c[num])
                score2.append(score2s)
            
        if a1[num]==2:
            outputs.append(torch.cat((out_c[num+1],output),-2))
        else : 
            outputs.append(output)
            
        return outputs,score2

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model,nhead, dim_feedforward=384*4, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.en_de_attn = En_De_Attention(d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def forward(self, tgt, memory, a1, score_c, out_c):
        tgt2,score2 = self.en_de_attn(tgt, memory,score_c,a1,out_c)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt,score2
      
        
        
class En_De_Attention(nn.Module):
    
    def __init__(self, d_model):
        super(En_De_Attention, self).__init__()
        
        # key, query, value projections for all heads
        
        
        self.known = nn.Linear(d_model, d_model)
        self.unknown = nn.Linear(d_model, d_model)
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(256, d_model)
        self.value = nn.Linear(256, d_model)
        self.attn_drop = nn.Dropout(0.0)
        self.n_head = 1

    def forward(self, x, y,score_c,tt,out_c):
        B, T, C = x.size()
        _, T1, _ = y.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(y).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(y).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        score = att
        
        if tt >0 and tt <= 1:
            unknow = self.unknown(x).view(B, T, C)
            know = self.known(out_c).view(B, -1, C)
            score1 = (unknow @ know.transpose(-2, -1)) * (1.0 / math.sqrt(know.size(-1)))
            score = tt * F.softmax((score1.unsqueeze(1) @ score_c), dim=-1) + (1-tt) * F.softmax(score, dim=-1)
            att=score
        else:
            att = F.softmax(score, dim=-1)
        att = self.attn_drop(att)
        z = att @ v
        z = z.transpose(1, 2).contiguous().view(B, T, C)  
        return z, score
