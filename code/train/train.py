""" Trainer for meta-train phase. """
import json
import os
import os.path as osp
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

#import plot
from dataset_loader import DatasetLoader as Dataset
from transformer_64 import Generator
from loss import PerceptualLoss, StyleLoss
from itertools import cycle
import random
class Trainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = '/DGTS/logs/'
        meta_base_dir = osp.join(log_base_dir, args.file_name)
        self.save_path = meta_base_dir
        if os.path.exists(self.save_path):
            pass
        else:
            os.makedirs(self.save_path)
        self.args = args
        ### data
        self.trainset = Dataset('train', self.args)
        self.train_loader = None
        ####### model #######
        self.netG = Generator().to(self.args.device)
        self.mask = torch.ones(self.args.batch_size, 1, self.args.image_size, self.args.image_size, device = self.args.device)
        self.mask[:, :, int((self.args.image_size - self.args.crop_size)//2): int((self.args.image_size + self.args.crop_size)//2), 
        int((self.args.image_size - self.args.crop_size)//2): int((self.args.image_size + self.args.crop_size)//2)] = 0.0
        
        
        self.perceptual_loss = PerceptualLoss().to(self.args.device)
        self.style_loss = StyleLoss().to(self.args.device)
        self.l1_loss = nn.L1Loss()
        
        #'''
        param_dicts = [
            {"params": [p for n, p in self.netG.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.netG.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
        ]
        self.optimizer_g = torch.optim.AdamW(param_dicts, lr=1e-4,weight_decay=1e-4)


        self.netG = torch.nn.DataParallel(self.netG)
        
        ## transform for mask
        self.transform = transforms.Compose([
        	transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
        	transforms.ToTensor(),
        ]) 

    def train(self):

        self.netG.train()
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=True) 
        for epoch in range(self.args.start_epoch, self.args.max_epoch + 1):
            for data_in in self.train_loader:
                
                real = data_in.to(self.args.device)
                B,C,H,W = real.size()

                #irrgular mask
                tmp = random.sample(range(0,12000),1)
                THE_PATH = osp.join('/DGTS/data/mask','%05d.png'%tmp[0])
                mask_in = self.transform(Image.open(THE_PATH).convert('1')).to(self.args.device)
                mask = mask_in.resize(1,1,H,W)
                mask = torch.repeat_interleave(1-mask, repeats=B, dim=0)
                
                #rgular mask
                #mask = self.mask
                
                real1 = F.interpolate(real, scale_factor = 0.25)
                mask1 = F.interpolate(mask, scale_factor =0.25)
                fakes = self.netG(real,mask)
                fake3 = fakes * (1. - mask1) + mask1 * real1

                loss1 = self.l1_loss(fake3 * (1. - mask1), real1 * (1. - mask1))/ (1-mask1).mean()
                loss2 = self.perceptual_loss(fake3, real1)
                loss3 = self.style_loss(fake3 * (1. - mask1), real1 * (1. - mask1))
                lossa = loss1  * 10 + loss2 * 0.1 + loss3 * 250
                
                self.optimizer_g.zero_grad()
                lossa.backward()
                self.optimizer_g.step()

            torch.save(self.netG.state_dict(), os.path.join(self.save_path,'Generator_{}.pth'.format(int(epoch))))