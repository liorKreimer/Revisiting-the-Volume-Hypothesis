#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataclasses import dataclass

@dataclass
class Params:
    n_train: int
    n_test: int    
    bins_e: int
    bins_q: int
    

n_train = 16 # number of training samples
n_test = 2000 # number of test samples
    




bins_q = 200   # test accuracy bins
'''
The accuracy is qacc in  [0,1,...n_test] 
The index of the q bin is
q = int(((qacc-1)/n_test)*bins_q)  in [0,1,...bins_q-1]
'''



bins_e = n_train+1    # training accuracy bins
'''
The accuracy is eacc in  [0,1,...n_train] 
If granularity is not maximal (i.e. if bins_e  < ntrain_+1), the index of the e bin is
e = int((eacc/n_train)*(bins_e-1))  in [0,1...bins_e-1]
Note that only when eacc==n_train_samples we get e == bins_e-1 (the maximal number)
'''

params = Params(n_train, n_test, bins_e, bins_q)

bins_q = 200   # test accuracy bins
range_bins_q = 100  # how many bins_q we are using

qw = 6  # number of windows in the q axis
#overlap = 0.5 
L = int((range_bins_q*2)/(qw+1))


eq_limits = {}
for w in range(qw):
    eq_limits[w] = {} 

    eq_limits[w]['q_min'] = 200 + int(w*L/2)
    eq_limits[w]['q_max'] = 200 + int(w*L/2 + L-1)
    # if w == qw-1:
    #     eq_limits[w]['q_max'] = bins_q -1
    
    eq_limits[w]['e_min'] = n_train -5
    eq_limits[w]['e_max'] = n_train
    



paired_ranks_for_exchange = {}
paired_ranks_for_exchange[0] = {}
paired_ranks_for_exchange[1] = {}

for i in range(0,qw,2):
    paired_ranks_for_exchange[0][i] = i+1
    paired_ranks_for_exchange[0][i+1] = i
    
paired_ranks_for_exchange[1][0] = -1
for i in range(1,qw-2,2):
    paired_ranks_for_exchange[1][i] = i+1
    paired_ranks_for_exchange[1][i+1] = i
paired_ranks_for_exchange[1][qw-1] = -1
    
    








# pairs_for_exchange = {}
# pairs_for_exchange[0] = {0:1, 1:0}
# pairs_for_exchange[1] = {1:2, 2:1}
# pairs_for_exchange[2] = {2:3, 3:2}
# pairs_for_exchange[3] = {3:4, 4:3}
# pairs_for_exchange[4] = {4:5, 5:4}
# pairs_for_exchange[5] = {6:7, 7:6}
# pairs_for_exchange[6] = {5:6, 6:5}




