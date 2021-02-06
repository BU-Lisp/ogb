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
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--sample_fraction', type=float, default=0.5, help='fraction to take which are tails of the relation')
    parser.add_argument('--write_train_totals', action='store_true')
    parser.add_argument('-u', '--use_testset_negatives', action='store_true')
    parser.add_argument('-F', '--Fmodel', action='store_true')
    parser.add_argument('--Fmodel_separate', action='store_true')

    return parser.parse_args(args)

args = parse_args()
k = args.k

np.set_printoptions( linewidth=150, precision=3, suppress=True )

def read_file(file):
    with open(file,"r") as input:
        return [int(n) for n in input.read().split()]
    
data = [ read_file(file) for file in args.files ]

data_array = np.array( data, dtype=float )

print( 'data array', data_array.shape )

nonzero_items = np.sum(data_array,axis=0)
print( '(#models): (#items)', np.unique(nonzero_items,return_counts=True) )

u,s,vt = svds( data_array, k=min(k,len(args.files)) )

print( 'input type:', re.sub('.*/', '', args.files[0] ) )

    
print('factors:', s)
print('v mean', np.mean(vt,axis=1))
print('v sd', np.std(vt,axis=1))
print('u*100:')
for i in range(len(args.files)):
    print( re.sub( '-extra/hits1.(head|tail)-batch.txt', '', args.files[i] ), np.mean(data_array,axis=1)[i], u[i,:]*100)


# compare to randomly shuffled

#print( data_array[:,0:4] )

for i in range(data_array.shape[1]):
    np.random.shuffle(data_array[:,i])

#print( data_array[:,0:4] )

u,s,vt = svds( data_array, k=min(k,len(args.files)) )

print('random s', s)
print('random u*100 mean', np.mean(u*100,axis=0))
print('random u*100 sd', np.std(u*100,axis=0))
print('random v mean', np.mean(vt,axis=1))
print('random v sd', np.std(vt,axis=1))
