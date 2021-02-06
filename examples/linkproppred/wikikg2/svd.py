'''
Read many vectors and compute svd
'''

import pandas as pd
import shutil, os, string, re
import os.path as osp
import torch
import numpy as np
import random
import pickle
import argparse
from scipy.sparse.linalg import svds

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Estimate Knowledge Graph stats',
        usage='estimator.py [<args>] [-h | --help]'
    )
    parser.add_argument('files', nargs='+', type=str)
    parser.add_argument('-m', '--mode', type=str, help='svd')
    parser.add_argument('--split', type=str, default='time')
    parser.add_argument('--newsplit', type=str)
    parser.add_argument('--maxN', type=int, default=500)
    parser.add_argument('--sample_fraction', type=float, default=0.5, help='fraction to take which are tails of the relation')
    parser.add_argument('--write_train_totals', action='store_true')
    parser.add_argument('-u', '--use_testset_negatives', action='store_true')
    parser.add_argument('-F', '--Fmodel', action='store_true')
    parser.add_argument('--Fmodel_separate', action='store_true')

    return parser.parse_args(args)

args = parse_args()

np.set_printoptions( linewidth=120, precision=4 )

def read_file(file):
    with open(file,"r") as input:
        return [int(n) for n in input.read().split()]
    
data = [ read_file(file) for file in args.files ]

data_array = np.array( data, dtype=float )

print( 'data array', data_array.shape )

u,s,vt = svds( data_array, k=min(10,len(args.files)) )

print( 'input type:', re.sub('.*/', '', args.files[0] ) )

for f in args.files:
    print( re.sub( '-extra/hits1.(head|tail)-batch.txt', '', f ) )
    
print('s', s)
print('u', u)

# compare to randomly shuffled

print( data_array[range(5),range(5)] )

for i in range(data_array.shape[1]):
    np.random.shuffle(data_array[:,i])

print( data_array[range(5),range(5)] )

u,s,vt = svds( data_array, k=min(10,len(args.files)) )

print('random s', s)

