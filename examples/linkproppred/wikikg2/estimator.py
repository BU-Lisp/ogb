'''
Directly compute simple statistics from the test set.
'''

import pandas as pd
import shutil, os, string, re
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
        description='Estimate Knowledge Graph stats',
        usage='estimator.py [<args>] [-h | --help]'
    )
    parser.add_argument('dataset', type=str)
    parser.add_argument('-m', '--mode', type=str, help='make_negs,check_negs,Fmodel,est_Fmodel')
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

make_data = True
newsplit = args.newsplit

if os.path.exists(args.dataset+'/split/'+args.split):
    data_in = args.dataset+'/split/'+args.split
else:
    meta = 'dataset_' + re.sub('-','_',args.dataset) + '/meta_dict.pt'
    if os.path.exists(meta):
        meta_dict = torch.load(meta)
        data_in = meta_dict['dir_path'] +'/split/'+args.split
    else:    
        data_in = 'dataset_' + args.dataset +'/split/'+args.split
print( 'read from', data_in )

fract = args.sample_fraction
extra = 1.1 # extra sample so that we can exclude triples in the graph
maxN = args.maxN

if os.path.isfile(data_in+'/split_dict.pt'):
    d = torch.load(data_in+'/split_dict.pt')
    train = d['train']
    valid = d['valid']
    test  = d['test']
else:
    train = torch.load(data_in+'/train.pt' )
    valid = torch.load(data_in+'/valid.pt' )
    test  = torch.load(data_in+'/test.pt' )

all = set()

nentity = max(np.max(train['head']),np.max(train['tail']))
nrelation = np.max(train['relation'])
print( 'nentity=%d nrelation=%d' % (nentity, nrelation) )

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

def triples(l):
    for t in l:
        for i in range(t['head'].shape[0]):
            yield ( t['head'][i], t['relation'][i], t['tail'][i] )

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


# estimate the "F" model

if args.mode == 'est_Fmodel':
    from scipy.sparse.linalg import svds
    counts = np.zeros( (nentity+1,2*nrelation+2), dtype=np.single )
    for h,r,t in triples([train]):
        counts[h,r] += 1
        counts[t,r+nrelation] += 1
    counts = counts / (np.linalg.norm( counts, ord=1, axis=0, keepdims=True )+1)
    if args.Fmodel_separate:
        counts_sep = np.swapaxes( counts.reshape( nentity+1, 2, nrelation+1 ), 1, 2 )
        u_s_vt = [svds(counts_sep[:,:,i],k=min(maxN,counts.shape[0],counts_sep.shape[1])-1) for i in (0,1)]
        u = numpy.concatenate( u_s_vt[0][0], u_s_vt[1][0], axis=1 )
        r_emb = numpy.concatenate( np.matmul( u_s_vt[0][2].transpose(), np.diag(u_s_vt[0][1]) ),
                                   np.matmul( u_s_vt[1][2].transpose(), np.diag(u_s_vt[1][1]) ), axis=1 )
    else:
        u, s, vt = svds(counts,k=min(maxN,counts.shape[0],counts.shape[1])-1)
        r_emb = np.matmul( vt.transpose(), np.diag(s) )
    print( 'singular values:', s )
    np.save( 'entity_embedding', u )
    np.save( 'relation_embedding', r_emb )
    exit( 0 )

    # sort by reverse F model scores
if args.Fmodel:
    u = np.load( 'entity_embedding.npy' )
    v = np.load( 'relation_embedding.npy' )
    fkt = {'head':dict(), 'tail':dict()}
    rel_offset = {'head':0, 'tail':nrelation}

def eval_Fscores( f, r ):
    if not r in fkt[f]:
        score = np.dot( u, v[r+rel_offset[f],:] )
        ord = np.argsort(-score)
        fkt[f][r] = ord[range(maxN)].tolist()
#        print( f, r, fkt[f][r] )
    return fkt[f][r]

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
            hist = np.zeros(maxN+1)
            for i in range(test[f+'_neg'].shape[0]):
                r = test['relation'][i]
                s = set(test[f+'_neg'][i]) & tset[r]
                tnegset.append(( r, s ))
                print( f, i, r, s, file=out )
                hist[len(s)] += 1
                if i % 100 == 0:
                    print( i, r, len(s) )
            print( f, hist )
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
# we can compute the true rank, or rank within a specified test set

hits = {'head':np.zeros(maxN+1),'tail':np.zeros(maxN+1)}
MRRsum = {'head':0, 'tail':0}
absent = {'head':0, 'tail':0}
present = {'head':0, 'tail':0}
variants = ['']

def init_stats(v):
    variants.append(v)
    for f in ('head','tail'):
        hits[f+v] = np.zeros(maxN+1)
        MRRsum[f+v] = 0
        present[f+v] = 0

if args.Fmodel:
    init_stats('_f')
if args.use_testset_negatives:
    init_stats('_set')

# what is correct rank given this info ????
# read in top 100 for each relation and give those scores

for i in range(test['head'].shape[0]):
    r = test['relation'][i]
    for f in ('head','tail'):
        n = len(ht[f].setdefault(r,[]))
        if n==0:
            absent[f] += 1
        else:
            if test[f][i] in hkt[f][r]:
                rank = hkt[f][r].index(test[f][i])
                present[f] += 1 
                if i % 100 == 1:
                    print( i, test[f][i], rank, hkt[f][r][:10] )
                for j in range(rank,maxN):
                    hits[f][j] += 1
                MRRsum[f] += 1.0/(1.0+rank)
                if args.use_testset_negatives: # not right ?
                    present[f+'_set'] += 1
                    testset = set(test[f+'_neg'][i])
                    common = testset & set(hkt[f][r])
                    newrank = len(testset & set(hkt[f][r][0:rank]))
                    print( 'old rank', rank, 'top testset elements', len(common), 'newrank', newrank )
                    for j in range(newrank,maxN):
                        hits[f+'_set'][j] += 1
                    MRRsum[f+'_set'] += 1.0/(1.0+newrank)
            if args.Fmodel:
                eval_Fscores( f, r )
                if test[f][i] in fkt[f][r]:
                    frank = fkt[f][r].index(test[f][i])
                    present[f+'_f'] += 1 
                    for j in range(frank,maxN):
                        hits[f+'_f'][j] += 1
                    MRRsum[f+'_f'] += 1.0/(1.0+frank)


print( absent, present )                    
for f in ('head','tail'):
    print( f, 'estHits1=', estHits1[f] )
    for s in variants:
        fs = f + s
        print( fs, 'MRR=', MRRsum[fs]/present[f] )
        print( fs, np.array2string( hits[fs][:10]/present[f], precision=8, threshold=np.inf, max_line_width=np.inf ) )

for s in variants:
    for N in (1,3,10):
        print( 'Test%s Hits@%d = %f' % (s, N, (hits['head'+s][(N-1)]+hits['tail'+s][(N-1)])/(present['head']+present['tail'])) )
    print( 'Test%s MRR = %f' % (s, (MRRsum['head'+s]+MRRsum['tail'+s])/(present['head']+present['tail'])) )
