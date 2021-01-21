'''
Directly compute simple statistics from the test set.
'''

import pandas as pd
import shutil, os, string
import os.path as osp
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw, read_binary_graph_raw, read_binary_heterograph_raw
import torch
import numpy as np
import random
import pickle

make_data = True
data_in = 'dataset/ogbl_wikikg2'
newsplit = 'byrel'

fract = 0.5 # fraction to take which are tails of the relation
extra = 1.1 # extra sample so that we can exclude triples in the graph
maxN = 500

if make_data:
    train = torch.load(data_in+'/split/time/train.pt' )
    valid = torch.load(data_in+'/split/time/valid.pt' )
test  = torch.load(data_in+'/split/time/test.pt' )

print( 'read in' )

heads = dict()
tails = dict()
ht = {'head':heads, 'tail':tails}
# at = {'head':train['head'].tolist(), 'tail':train['tail'].tolist()}
all = set()


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

if make_data:
    for h,r,t in triples([train,valid,test]):
        hd = heads.setdefault(r,{})
        hd[h] = hd.setdefault(h,0) + 1
        td = tails.setdefault(r,{})
        td[t] = td.setdefault(t,0) + 1
        all.add((h,r,t))

if 0:
    with open('est','w') as out:
        for r in heads.keys():
            print( r, htotals[r], ttotals[r], heads[r], tails[r], file=out )
    
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

htotals = {h: sum(heads[h].values()) for h in heads.keys()}
ttotals = {t: sum(tails[t].values()) for t in tails.keys()}

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
for f in ('head','tail'):
    print( f, np.array2string( hits[f], precision=8, threshold=np.inf, max_line_width=np.inf ) )
for f in ('head','tail'):
    print( f, np.array2string( hits[f]/present[f], precision=8, threshold=np.inf, max_line_width=np.inf ) )
