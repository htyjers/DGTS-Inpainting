""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        # Set the path according to train, val and test   
        self.args = args
        THE_PATH = None     
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train')
        elif setname=='test':
            THE_PATH = osp.join(args.dataset_dir, 'test')
        else:
            raise ValueError('Wrong setname.') 
       
        data = []
        for root, dirs, files in os.walk(THE_PATH, topdown=True):
            for name in files:
                data.append(osp.join(root, name))
                
        self.data = data    
        print(len(self.data))
        self.image_size = args.image_size

        self.transform = transforms.Compose([
          transforms.Resize((self.image_size,self.image_size)),
          transforms.ToTensor() ,
          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image
