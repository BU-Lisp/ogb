'''
Directly compute simple statistics from the test set, i.e. P(tail|rel) model.
Can also estimate numbers of motifs, by the following algorithm: sample A->r B, look for B->s C and A->t C.
'''

import pandas as pd
import shutil, os, string
import os.path as osp
#from ogb.utils.url import decide_download, download_url, extract_zip
#from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw, read_binary_graph_raw, read_binary_heterograph_raw
import torch
import numpy as np
import random
import pickle
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Convert Knowledge Graph Formats',
        usage='convertkg.py [<args>] [-h | --help]'
    )
    parser.add_argument('dataset', type=str)
    parser.add_argument('-m', '--mode', type=str, help='make_negs,check_negs,count_motifs')
    parser.add_argument('--split', type=str, default='time')
    parser.add_argument('--newsplit', type=str)
    parser.add_argument('-N', '--maxN', type=int, default=500)
    parser.add_argument('--sample_fraction', type=float, default=0.5, help='fraction to take which are tails of the relation')
    parser.add_argument('--write_train_totals', action='store_true')

 
    return parser.parse_args(args)

args = parse_args()

make_data = True
data_in = 'dataset/' + args.dataset
newsplit = args.newsplit

fract = args.sample_fraction
extra = 1.1 # extra sample so that we can exclude triples in the graph
maxN = args.maxN
max_motifs_per_edge = 1000

if make_data:
    train = torch.load(data_in+'/split/'+args.split+'/train.pt' )
    valid = torch.load(data_in+'/split/'+args.split+'/valid.pt' )
test  = torch.load(data_in+'/split/'+args.split+'/test.pt' )

all = set()

def some_triples(t,sample):
    for i in sample:
        yield ( t['head'][i], t['relation'][i], t['tail'][i] )
    
def triples(l):
    for t in l:
        for trip in some_triples( t, range(t['head'].shape[0]) ):
            yield trip

# given a list of edges, find all triangle motifs in which it is the first edge
# edge_table is sets of tails indexed by head, rel_table is rels indexed by both
def list_triangles(edge_table,rel_table,edges):
    for (h,r,t) in edges:
        i = 0
        if not h in edge_table:
            print( 'head not in edge_table', h, r, t )
        if t in edge_table:
            for third in list(edge_table[h] & edge_table[t]):
                for r1 in rel_table[(h,third)]:
                    for r2 in rel_table[(t,third)]:
                        yield [ ( h, t, third ), ( r, r1, r2 ) ]
                        i += 1
                if i==0:
                    print( 'did not find in rel_table', h, r, t, third )
        i = min( i, max_motifs_per_edge )
        motifs_per_edge_histogram[i] += 1        

def build_edge_rel_table( l ):
    et = dict()
    rt = dict()
    i = 0
    for h,r,t in triples(l):
        eth = et.setdefault(h,set())
        eth.add( t )
        rtht = rt.setdefault((h,t),[])
        rtht.append( r )
        i += 1
        if False and i % 10000 == 0:
            print( i, h, r, t, eth, rtht )
    return (et, rt)

if args.mode == 'count_motifs':
    (edge_table, rel_table) = build_edge_rel_table( [train] )
    print( 'Build edge and relation tables...' )
    motifs_per_edge_histogram = np.zeros(max_motifs_per_edge)
    sample = random.sample( range(train['head'].shape[0]), args.maxN )
    triangles = []
    with open(dataset+'motifs.txt','w') as out:
        i = 0
        for tri in list_triangles( edge_table, rel_table, some_triples( train, sample ) ):
            triangles.append(tri[1])
            print(tri,file=out)
            i += 1
            if i % 1000 == 0:
                print( i, tri )
    print( np.trim_zeros(motifs_per_edge_histogram,'b') )
    np.savez(args.dataset+'motifs', motifs=np.array(triangles) )
    exit(0)

# return N samples of field f for relation r
def sample(N,f,r,v,ex=extra):
    Nr = int(N * fract)
    Nt = int(N * ex)
    if Nr < len(ht[f][r]):
        sl = random.sample( ht[f][r], Nr )
    else:
        sl = ht[f][r]
    sl = sl + random.sample( at[f], Nt-len(sl) )
    # remove those for which (x,r,v) or (v,r,x) is in the graph
    if f=='head':
        sl = [x for x in sl if (x,r,v) not in all]
    else:
        sl = [x for x in sl if (v,r,x) not in all]
    # remove duplicates
    sl = list(set(sl))
    if len(sl) < N:
        print( 'error: too few negs', len(sl), 'for item', f, r, v, 'try again' )
        sl = sample(N,f,r,v,ex*ex*N/(len(sl)+1))
    return sl[:N]

# the dict "heads" is a list of all head frequencies indexed by relation (resp tails)
def make_tables( l ):
    heads = dict()
    tails = dict()
    for h,r,t in triples(l):
        hd = heads.setdefault(r,{})
        hd[h] = hd.setdefault(h,0) + 1
        td = tails.setdefault(r,{})
        td[t] = td.setdefault(t,0) + 1
        all.add((h,r,t))
    htotals = {h: sum(heads[h].values()) for h in heads.keys()}
    ttotals = {t: sum(tails[t].values()) for t in tails.keys()}
    return { 'head':heads, 'tail':tails, 'htotals':htotals, 'ttotals':ttotals }

train_tab = make_tables( [train] )
test_tab = make_tables( [test] )
heads = train_tab['head']
tails = train_tab['tail']
htotals = train_tab['htotals']
ttotals = train_tab['ttotals']

if args.write_train_totals:
    with open('est','w') as out:
        for r in heads.keys():
            print( r, htotals[r], ttotals[r], heads[r], tails[r], file=out )
    exit(0)
    
# sort the dictionaries by reverse frequency

headkeys = dict()
tailkeys = dict()
if make_data:
    for r in heads.keys():
        headkeys[r] = sorted(heads[r].keys(), key=heads[r].__getitem__, reverse=True)
        del headkeys[r][maxN:]
        tailkeys[r] = sorted(tails[r].keys(), key=tails[r].__getitem__, reverse=True)
        del tailkeys[r][maxN:]
#    with open('est.data','w') as out:
#        pickle.dump( [ heads, tails, headkeys, tailkeys ], out )
else:
    with open( 'est.data' ) as f:
        ( heads, tails, headkeys, tailkeys ) = pickle.load( f )

hkt = {'head':headkeys, 'tail':tailkeys}
ht = {'head':heads, 'tail':tails}
htot = {'head':htotals, 'tail':ttotals}
testhtot = {'head':test_tab['htotals'], 'tail':test_tab['ttotals']}
estHits1 = dict()
P1 = dict()
P1w = dict()

# for each test item, count number of negs with correct relation type
if args.mode == 'check_negs':
    with open( 'check.out', 'w' ) as out:
        for f in ('head','tail'):
            tset = { r: set(train_tab[f]) for r in train_tab[f].keys() }
            tnegset = []
            hist = np.zeros(maxN+1,dtype=int)
            for i in range(test[f+'_neg'].shape[0]):
                r = test['relation'][i]
                s = set(test[f+'_neg'][i]) & tset[r]
                tnegset.append(( r, s ))
                print( f, i, r, s, file=out )
                hist[len(s)] += 1
            print( f, np.trim_zeros(hist,'b') )
    exit(0)


if args.mode != 'test_negs':
    raise ValueError('unknown mode', args.mode)

    
# estimate Hits@1 by relation, it is just the weighted mean of P_1.

with open( 'est.out', 'w' ) as out:
    print( 'subst relation Ntrain Ntest topv P1 P1w', file=out )
    for f in ('head','tail'):
        P1[f] = { r: float(ht[f][r][hkt[f][r][0]])/htot[f][r] for r in htot[f].keys() }
        P1w[f] = { r: P1[f][r] * testhtot[f][r] /test[f].shape[0] for r in testhtot[f].keys() if r in P1[f].keys() }
        estHits1[f] = sum( P1w[f].values() )
        for r in P1w[f].keys():
            print( f, r, htot[f][r], testhtot[f][r], hkt[f][r][0], P1[f][r], P1w[f][r], file=out )


# estimate Hits@N and MRR for the testset

hits = {'head':np.zeros(maxN+1),'tail':np.zeros(maxN+1)}
MRRsum = {'head':0, 'tail':0}
absent = {'head':0, 'tail':0}
present = {'head':0, 'tail':0}

for i in range(test['head'].shape[0]):
    r = test['relation'][i]
    for f in ('head','tail'):
        n = len(ht[f].setdefault(r,[]))
        if n==0:
            absent[f] += 1
        else:
            present[f] += 1 
            if test[f][i] in hkt[f][r]:
                rank = hkt[f][r].index(test[f][i])
                if i % 100 == 1:
                    print( i, test[f][i], rank, hkt[f][r][:10] )
                for j in range(rank,maxN):
                    hits[f][j] += 1
                MRRsum[f] += 1.0/(1.0+rank)

for f in ('head','tail'):
    print( f, 'absent=', absent[f], 'present=', present[f], 'MRR=', MRRsum[f]/present[f] )
    print( f, np.array2string( hits[f][:10]/present[f], precision=8, threshold=np.inf, max_line_width=np.inf ) )
    print( f, 'estHits1=', estHits1[f] )
