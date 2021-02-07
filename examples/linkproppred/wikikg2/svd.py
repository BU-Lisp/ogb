'''
Read many vectors and compute svd
'''

import pandas as pd
import shutil, os, string, re, sys
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
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('--print_v0', action='store_true')
    parser.add_argument('--split', type=str, default='time')
    parser.add_argument('--newsplit', type=str)
    parser.add_argument('--sample_fraction', type=float, default=0.5, help='fraction to take which are tails of the relation')
    parser.add_argument('--write_train_totals', action='store_true')
    parser.add_argument('-u', '--use_testset_negatives', action='store_true')
    parser.add_argument('--Fmodel_separate', action='store_true')

    return parser.parse_args(args)

def analyze( arr, k, label='' ):
    u,s_all,vt = svds( arr, k=k )
    s_sum = np.sum(s_all)
    ind = np.argsort(-s_all)[0:k]
    perf_all = s_all * np.mean(vt,axis=1) * 100
    u,s,vt,perf = u[:,ind], s_all[ind], vt[:,ind], perf_all[ind]
    print(label, 'factors', np.sum(s), s)
    print(label, 'perf', np.sum(perf_all), perf)
    if label!='random':
        if args.print_v0:
            print( vt[:s,0] )
        for i in range(len(args.files)):
            print( label, re.sub( '-extra/hits1.(head|tail)-batch.txt', '', args.files[i] ), ('%2.2f' % (np.mean(data_array,axis=1)[i]*100)), u[i,:]*perf)
    return s_all

args = parse_args()
k = min( args.k+1,len(args.files) )-1

print( 'input type:', re.sub('.*/', '', args.files[0] ) )

np.set_printoptions( linewidth=150, precision=3, suppress=True, threshold=sys.maxsize )

def read_file(file):
    with open(file,"r") as input:
        return [int(n) for n in input.read().split()]
    
data = [ read_file(file) for file in args.files ]

data_array = np.array( data, dtype=float )

print( 'data array', data_array.shape )

if args.hist:
    nonzero_items = np.sum(data_array,axis=0)
    print( '(#models): (#items)', np.unique(nonzero_items,return_counts=True) )

s_all = analyze( data_array, k, '' )

# compare to randomly shuffled

for i in range(data_array.shape[1]):
    np.random.shuffle(data_array[:,i])

s_rand = analyze( data_array, k, 'random' )

print( np.count_nonzero(s_all > s_rand[1]), 'significant factors' ) # should use TW dist
