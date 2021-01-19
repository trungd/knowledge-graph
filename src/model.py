from typing import Dict, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.torch import BaseModel
from dlex.utils import logging, logger

from torch.utils.data import DataLoader
# from dataloader import TestDataset
from collections import defaultdict


class Model(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        self.nentity = dataset.nentity
        self.nrelation = dataset.nrelation
        self.hidden_dim = self.configs.hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(torch.Tensor([self.configs.gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = self.hidden_dim * 2 if self.configs.double_entity_embedding else self.hidden_dim
        self.relation_dim = self.hidden_dim * 2 if self.configs.double_relation_embedding else self.hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(self.nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.evaluator = dataset.evaluator

    def score_func(self, head, relation, tail, mode):
        return None

    def get_score(self, sample, mode='single'):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == 'single':
            head = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        return self.score_func(head, relation, tail, mode)

    def forward(self, batch):
        negative_score = self.get_score((batch.positive_sample, batch.negative_sample), mode=batch.mode)
        if self.configs.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.configs.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = self.get_score(batch.positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        return positive_score, negative_score

    def get_loss(self, batch, scores):
        positive_score, negative_score = scores

        if self.configs.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(batch.subsampling_weight * positive_score).sum() / batch.subsampling_weight.sum()
            negative_sample_loss = -(batch.subsampling_weight * negative_score).sum() / batch.subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if self.configs.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = self.configs.regularization * (
                    self.model.entity_embedding.norm(p=3) ** 3 +
                    self.model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization

        return loss

    def infer(self, batch):
        scores = self.get_score((batch.positive_sample, batch.negative_sample), mode=batch.mode)
        return [scores], batch, None, None


class TransE(Model):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def score_func(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


class PairRE(Model):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def score_func(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


class DistMult(Model):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def score_func(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score


class ComplEx(Model):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        if not self.configs.double_entity_embedding or not self.configs.double_relation_embedding:
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def score_func(self, head, relation, tail, mode):
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

        score = score.sum(dim=2)
        return score


class RotatE(Model):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        if not self.configs.double_entity_embedding or self.configs.double_relation_embedding:
            raise ValueError('RotatE should use --double_entity_embedding')

    def score_func(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

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

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score