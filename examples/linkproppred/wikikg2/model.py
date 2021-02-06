#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator

def extra_test_statistics(input_dict):
    mode, y_pred_pos, y_pred_neg = input_dict['mode'], input_dict['y_pred_pos'], input_dict['y_pred_neg']
    positive_sample = input_dict['positive_sample']
    negative_sample = input_dict['negative_sample']

    y_pred = torch.cat([y_pred_pos.view(-1,1), y_pred_neg], dim = 1)
    argsort = torch.argsort(y_pred, dim = 1, descending = True)
    ranking_list = (argsort == 0).nonzero(as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    toparg = argsort[:,0]
    nextarg = argsort[:,1]
    y_neg_mean = torch.mean(y_pred_neg,1)
    y_neg_sd = torch.std(y_pred_neg,1)
    print( 'mode', 'head', 'relation', 'tail', 'neg1', 'neg2', 'score', 'rank', 'topscore', 'toparg', 'mean', 'sd', 'nextscore', 'nextarg' )
    for i in range(len(ranking_list)):
        print( mode, positive_sample[i,0].item(), positive_sample[i,1].item(), positive_sample[i,2].item(),
               negative_sample[i,toparg[i]].item(), negative_sample[i,nextarg[i]].item(),
               y_pred_pos[i].item(), ranking_list[i].item(), 
               y_pred[i,toparg[i].item()].item(), toparg[i].item(),
               y_neg_mean[i].item(), y_neg_sd[i].item(),
               y_pred[i,nextarg[i].item()].item(), nextarg[i].item() )


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        (model_name,count_FR) = re.subn('FR$','',model_name)
        (model_name,count_XR) = re.subn('XR$','',model_name)
        self.model_name = model_name
        self.nentity = nentity
        if count_XR > 0:
            self.Mrelations = 0
            nrelation = 1
        else:
            self.Mrelations = 1
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter( torch.zeros(nrelation, self.relation_dim), requires_grad = count_FR==0 )
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item(),
        )
        
        if model_name in ['TuckER', 'Groups']:
            self.tensor_weights = nn.Parameter(
                torch.zeros(self.hidden_dim,self.hidden_dim,self.hidden_dim)) # head x tail x rel
            nn.init.uniform_(
                tensor=self.tensor_weights,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )


        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['Base', 'TransE', 'DistMult', 'ComplEx', 'RotatE', 'PairRE', 'TuckER', 'Groups', 'F', 'RE', 'NetRE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name in ['RotatE','Groups'] and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name in ['ComplEx','NetRE'] and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name in ['PairRE','F','RE'] and not double_relation_embedding:
            raise ValueError('PairRE and F/RE should use --double_relation_embedding')

        self.evaluator = evaluator

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1] * self.Mrelations
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1] * self.Mrelations
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1] * self.Mrelations
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'BasE': self.BasE,
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'PairRE': self.PairRE,
            'TuckER': self.TuckER,
            'Groups': self.Groups,
            'F': self.F,
            'RE': self.F,
            'NetRE': self.NetRE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def BasE(self, head, relation, tail, mode):
        score = head - tail
        score = torch.norm(score, p=self.pnorm, dim=2)
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def Groups(self, head, relation, tail, mode):
        head_h, head_t = torch.chunk(head, 2, dim=2)
        tail_h, tail_t = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            score = torch.einsum( 'htr,bnh,bit,bir->bn', self.tensor_weights, head_h, tail_t, relation )
        else:
            score = torch.einsum( 'htr,bih,bnt,bir->bn', self.tensor_weights, head_h, tail_t, relation )
        return self.gamma.item() - score

    def TuckER(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = torch.einsum( 'htr,bnh,bit,bir->bn', self.tensor_weights, head, tail, relation )
        else:
            score = torch.einsum( 'htr,bih,bnt,bir->bn', self.tensor_weights, head, tail, relation )
        return self.gamma.item() - score

    def F(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)
        if mode == 'head-batch':
            score = head * re_head
        else:
            score = tail * re_tail
        return torch.sum(score, dim=2)

    def NetRE(self, head, relation, tail, mode):
        head_n, head_r = torch.chunk(head, 2, dim=2)
        tail_n, tail_r = torch.chunk(tail, 2, dim=2)
        score1 = self.BasE(head_n, relation, tail_n, mode)
        score2 = self.F(head_r, relation, tail_r, mode)
        return score1 + score2


    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, args, random_sampling=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()

        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                args, 
                'head-batch',
                random_sampling
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                args, 
                'tail-batch',
                random_sampling
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)

                    batch_results = model.evaluator.eval({'y_pred_pos': score[:, 0], 
                                                'y_pred_neg': score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if args.extra_test_statistics:
                        extra_test_statistics({'mode': mode,
                                               'y_pred_pos': score[:, 0], 
                                               'y_pred_neg': score[:, 1:],
                                               'positive_sample': positive_sample,
                                               'negative_sample': negative_sample })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics
