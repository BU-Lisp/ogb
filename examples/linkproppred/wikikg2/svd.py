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
    parser.add_argument('--hist', action='store_true')
    parser.add_argument('--Fmodel_separate', action='store_true')

    return parser.parse_args(args)

def analyze( arr, k, label='' ):
    u,s,vt = svds( arr, k=k )
    s_sum = np.sum(s)
    ind = np.argsort(-s)[0:k]
    perf_all = s * np.sum(vt,axis=1) * 100
    u,s,vt,perf = u[:,ind], s[ind], vt[:,ind], perf_all[ind]
    print(label, 'factors', np.sum(s), s)
    print(label, 'perf', np.sum(perf_all), perf)
    for i in range(len(args.files)):
        print( label, re.sub( '-extra/hits1.(head|tail)-batch.txt', '', args.files[i] ), ('%2.2f' % (np.mean(data_array,axis=1)[i]*100)), u[i,:]*perf)
    

args = parse_args()
k = min( args.k,len(args.files) )-1

print( 'input type:', re.sub('.*/', '', args.files[0] ) )

np.set_printoptions( linewidth=150, precision=3, suppress=True )

def read_file(file):
    with open(file,"r") as input:
        return [int(n) for n in input.read().split()]
    
data = [ read_file(file) for file in args.files ]

data_array = np.array( data, dtype=float )

print( 'data array', data_array.shape )

if args.hist:
    nonzero_items = np.sum(data_array,axis=0)
    print( '(#models): (#items)', np.unique(nonzero_items,return_counts=True) )

analyze( data_array, k, '' )

# compare to randomly shuffled

for i in range(data_array.shape[1]):
    np.random.shuffle(data_array[:,i])

analyze( data_array, k, 'random' )
