""" Main function for this repo. """
import argparse
import numpy as np
import torch
from test import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic parameters 
    parser.add_argument('--num_work', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default=None) # Dataset folder
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--file_name', type=str, default='test')  ## set test Folder name
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test']) # Phase

    args = parser.parse_args()
    print(args)

    # Set manual seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    trainer = Trainer(args)
    trainer.train()
