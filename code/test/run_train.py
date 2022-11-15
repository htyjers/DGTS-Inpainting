""" Generate commands for meta-train phase. """
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
def run_exp():    
    the_command = (
        'python3 /DGTS/code/test/main.py' 
        + ' --dataset_dir=' + '/DGTS/data/places2'
        
    )

    os.system(the_command + ' --phase=test')

run_exp()
