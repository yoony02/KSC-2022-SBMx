#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018
Modified on January, 2022
@author: Tangrizzly
#modifier: heeyooon

function create randomwalk using weight 
=> refer https://github.com/stellargraph/stellargraph/blob/develop/stellargraph/data/explorer.py
"""

import networkx as nx
import numpy as np
import random
from six import iterkeys
from collections import defaultdict, Iterable
import pickle



def get_metric_scores(scores, targets, k, ht_dict, pop_dict, tail_idxs, eval):
    # eval : hit, mrr, cov, arp, tail, tailcov
    sub_scores = scores.topk(k)[1]
    sub_scores = sub_scores.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    cur_hits, cur_mrrs, cur_pops = [], [], []
    for score, target in zip(sub_scores, targets):
        isin = np.isin(score, target)
        if sum(isin) == 1:
            cur_hits.append(True)
            cur_mrrs.append(1 / (np.where(isin == True)[0][0] + 1))
        else:
            cur_hits.append(False)
            cur_mrrs.append(0)
        
        cur_pops.append(np.mean([pop_dict[item+1] for item in score]))
    
    # overall acc and coverage
    eval[0] += cur_hits
    eval[1] += cur_mrrs
    eval[2] += np.unique(sub_scores).tolist()

    # tail acc and cov
    eval[3] += np.array(cur_hits)[tail_idxs].tolist()
    eval[4] += np.array(cur_mrrs)[tail_idxs].tolist()
    eval[5] += [np.mean(np.sum(~np.isin(sub_scores, ht_dict['head']), axis=1) / k)]  # Tail@K
    eval[6] += np.unique(sub_scores[~np.where(np.isin(sub_scores, ht_dict['head']))[0]]).tolist() # Tail Coverage@K

    # entropy : head and tail
    # head_cnt_pct = (np.sum(np.isin(sub_scores, ht_dict['head']), axis=1) / k)
    # tail_cnt_pct = 1 - head_cnt_pct
    # entropy = -((head_cnt_pct) * np.log2(head_cnt_pct) + (tail_cnt_pct) * np.log2(tail_cnt_pct))
    # eval[7] += np.nan_to_num(entropy).tolist()

    # ARP
    eval[7] += cur_pops

    
    return eval


def metric_print(eval10, eval20, n_node, time):

    for evals in [eval10, eval20]:
        # hit, mrr, cov, arp, tail, tailcov
        evals[0] = np.mean(evals[0]) * 100
        evals[1] = np.mean(evals[1]) * 100
        evals[2] = len(np.unique(evals[2])) / n_node * 100
        evals[3] = np.mean(evals[3]) * 100
        evals[4] = np.mean(evals[4]) * 100
        evals[5] = np.mean(evals[5]) * 100
        evals[6] = len(np.unique(evals[6])) / n_node * 100
        evals[7] = np.mean(evals[7])

    # print('Metric\t\tHR@10\tMRR@10\tCov@10\tHRt@10\tMRRt@10\tTail@10\tTCov@10\tEntropy@10')
    print('Metric\t\tHR@10\tMRR@10\tCov@10\tHRt@10\tMRRt@10\tTail@10\tTCov@10\tARP@10')
    print(f'Value\t\t'+'\t'.join(format(eval, ".2f") for eval in eval10))

    # print('Metric\t\tHR@20\tMRR@20\tCov@20\tHRt@20\tMRRt@20\tTail@20\tTCov@20\tEntropy@20')
    print('Metric\t\tHR@20\tMRR@20\tCov@20\tHRt@20\tMRRt@20\tTail@20\tTCov@20\tARP@20')
    print(f'Value\t\t' + '\t'.join(format(eval, ".2f") for eval in eval20))

    print(f"Time elapse : {time}")
    return [eval10, eval20]


def get_best_result(results, epoch, best_results, best_epochs):
    # results: eval10, eval20
    # eval: HR, MRR, cov, tail, tailcov, HRtail, MRRtail

    for result, best_result, best_epoch in zip(results, best_results, best_epochs):
        flag = 0
        for i in range(8):           
            if result[i] > best_result[i]:
                best_result[i] = result[i]
                best_epoch[i] = epoch
                flag = 1

    print("-"*100)
    # print('Best Result\tHR@10\tMRR@10\tCov@10\tHRt@10\tMRRt@10\tTail@10\tTCov@10\tEntropy@10\tEpochs')
    print('Best Result\tHR@10\tMRR@10\tCov@10\tHRt@10\tMRRt@10\tTail@10\tTCov@10\tARP@10\tEpochs')
    print(f'Value\t\t' + '\t'.join(format(result, ".2f") for result in best_results[0])
          + '\t' + ', '.join(str(epoch) for epoch in best_epochs[0]))

    # print('Best Result\tHR@20\tMRR@20\tCov@20\tHRt@20\tMRRt@20\tTail@20\tTCov@20\tEntropy@20\tEpochs')
    print('Best Result\tHR@20\tMRR@20\tCov@20\tHRt@20\tMRRt@20\tTail@20\tTCov@20\tARP@20\tEpochs')
    print(f'Value\t\t' + '\t'.join(format(result, ".2f") for result in best_results[1])
          + '\t' + ', '.join(str(epoch) for epoch in best_epochs[1]))

    return flag


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, dataset, shuffle=False):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.dataset = dataset
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
    

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i, ht_dict):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        tail_idxs = np.where(np.isin(targets, ht_dict['tail']))[0].tolist()

        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        
        
        return alias_inputs, A, items, mask, targets-1, tail_idxs
        
        
