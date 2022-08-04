""" Generate commands for meta-train phase. """
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # 114
def run_exp():    
    the_command = (
        'python3 /DGTS/code/train/main.py' 
        + ' --dataset_dir=' + '/DGTS/data/places'
        
    )

    os.system(the_command + ' --phase=train')

run_exp()          ## best 
