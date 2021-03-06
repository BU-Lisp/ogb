'''
Analyze embeddings.
Can also relate to numbers of motifs.
To add: check entity embeddings to see how well they correspond to motif embeddings.
'''

import numpy as np
import networkx as nx
import argparse
from torch import load
import os, re, csv
import random

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Convert numpy saved data to text',
        usage='readnp.py [<file>] [-h | --help]'
    )

    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-pn', '--print_norms', action='store_true')
    parser.add_argument('-px', '--print_normsx', action='store_true')
    parser.add_argument('-P', '--PairRE', action='store_true')
    parser.add_argument('-t', '--tensor', action='store_true')
    parser.add_argument('-cht', '--compare_head_tail', action='store_true')
    parser.add_argument('-s', '--sample', type=float, default=1.0 )
    parser.add_argument('--motif_data', type=str, default='' )
    parser.add_argument('--random_motifs', type=int, default=0 )
    return parser.parse_args(args)

def split_head_tail(v):
    d = v.shape[1]
    return (v[:,0:(d//2)], v[:,(d//2):d])
    
def motif(v):
    if args.PairRE:
        (h, t) = split_head_tail(v)
        return h[0,:]*h[1,:]*t[2,:] - t[0,:]*t[1,:]*h[2,:]
    else:
        return v[0,:]+v[1,:]-v[2,:]

args = parse_args()

data = np.load(args.infile)

    
if args.motif_data!='':
    with open(args.outfile, 'w') as out:
        if args.random_motifs>0:
            z = np.zeros(args.random_motifs,dtype=float)
            for i in range(args.random_motifs):
                m = random.sample( range(data.shape[0]), 3 )
                z[i] = np.linalg.norm(motif(data[m,:]), ord=1)
                print( m, 'random', z[i], file=out )
            print( 'random mean=', np.mean(z), 'sd=', np.std(z) )
            mean_ran_z, sd_ran_z, n_ran_z = np.mean(z), np.std(z), args.random_motifs
        motif_npzfile = np.load(args.motif_data)
        print( 'read motif data containing', motif_npzfile.files )
        motifs = motif_npzfile['motifs']
        motif_count = dict()
        for m in motifs:
            mid = (m[0], m[1], m[2])
            motif_count[mid] = motif_count.setdefault(mid,0) + 1
        print( 'counted', sum(motif_count.values()), 'motifs,', len(motif_count), 'distinct' )
        z = np.zeros(len(motif_count.keys()),dtype=float)
        i = 0
        for m in motif_count.keys():
            if m[0]!=m[1] and m[1]!=m[2] and m[2]!=m[0]:
                z[i] = np.linalg.norm(motif(data[m,:]), ord=1)
                print( m, motif_count[m], z[i], file=out )
                i += 1
        z = np.trim_zeros(z,'b')
        print( 'motif mean=', np.mean(z), 'sd=', np.std(z) )
        mean_z, sd_z, n_z = np.mean(z), np.std(z), i
        if args.random_motifs>0:
            sdelta = np.sqrt( sd_ran_z*sd_ran_z/n_ran_z +  sd_z*sd_z/n_z )
            print( 't value=', (mean_ran_z-mean_z)/sdelta )
    exit(0)
    
if args.sample<1.0:
    n_orig = data.shape[0]
    select = np.random.sample(size=n_orig) < args.sample
    data = np.compress( select, data, axis=0 )
    print( 'sampled', data.shape[0], 'out of', n_orig )

if args.print_norms:
    with open(args.outfile, 'w') as out:
        print( 'norms:', np.linalg.norm(data, ord=1, axis=1).tolist(), file=out )
        print( 'motif:', np.linalg.norm(motif(data[1:4,:]), ord=1), file=out )    
        if args.print_normsx and data.shape[0]>4:
            print( 'motif2:', np.linalg.norm(motif(data[4:7,:]), ord=1), file=out )    
#        if args.print_pair:

elif args.compare_head_tail:
    (h, t) = split_head_tail(data)
    with open(args.outfile, 'w') as out:
        for i in range(data.shape[0]):
            print( i, '|h|=', np.linalg.norm(h[i,:],1), '|t|=', np.linalg.norm(t[i,:],1), '|h-t|=', np.linalg.norm(h[i,:]-t[i,:],1), file=out )

elif args.tensor:
    s = np.array2string( data, precision=8, threshold=np.inf, suppress_small=True )
    with open(args.outfile, 'w') as out:
        print( s, file=out )
else:
    np.savetxt(args.outfile, data, fmt='%.8e', header='Read from ' + args.infile)
