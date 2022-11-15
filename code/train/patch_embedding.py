
from torch.nn import Dropout,Conv2d
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
import torch
import numpy as np
import math
from position_encoding import PositionEmbeddingSine
import torch.nn.functional as F
from position_encoding import build_position_encoding
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, patch_size=8, img_size=64, in_channels=3, hidden_size = 384):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        self.patch_size = patch_size

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        
        weight_maskUpdater = torch.ones(1, 1,8, 8)
        self.weight_maskUpdater = torch.as_tensor(weight_maskUpdater)
        self.dropout = Dropout(0.1)
        
  
    def forward(self, x, mask,i=-1):
        
        if i == -1:
            pos1 = positionalencoding2d(384,8,8)
            pos = torch.tensor(pos1).clone().detach().cuda()
            x = self.patch_embeddings(x * mask) + pos
            x = x.flatten(2)
            x = x.transpose(-1, -2)
            embeddings = x 
            
            update_mask = F.conv2d(mask, self.weight_maskUpdater.cuda(), bias=None, stride=self.patch_size)
            
            update_mask1 = torch.clamp(update_mask, 48, 64)-48
            update_mask1 = update_mask1.flatten(2)
            update_mask1 = update_mask1.transpose(-1, -2).squeeze(2)
            
            update_mask2 = update_mask.flatten(2)
            update_mask2 = update_mask2.transpose(-1, -2).squeeze(2)
            
            update_mask = torch.clamp(update_mask, 60, 61)-60
            update_mask = update_mask.flatten(2)
            update_mask = update_mask.transpose(-1, -2).squeeze(2)
            return embeddings, update_mask, update_mask1, update_mask2
        else:
            x = self.patch_embeddings(x)
            return x

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
