from collections import defaultdict
import random
from typing import List

import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import LinkPropPredDataset, Evaluator
from tqdm import tqdm
import numpy as np

from dlex import Params, logger
from dlex.datasets import DatasetBuilder
from dlex.datasets.builder import ModelStringOutput
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.utils import logging


from ogb.linkproppred import Evaluator


class KGBatch(Batch):
    mode: str
    subsampling_weight: torch.Tensor
    negative_sample: torch.Tensor
    positive_sample: torch.Tensor

    def __len__(self):
        return self.positive_sample.shape[0]


def collate_fn(batch: List):
    batch = [b for b in batch if b is not None]
    positive_sample = torch.stack([_[0] for _ in batch], dim=0)
    negative_sample = torch.stack([_[1] for _ in batch], dim=0)
    if batch[0][2] is not None:
        subsampling_weight = torch.cat([_[2] for _ in batch], dim=0)
    else:
        subsampling_weight = None
    mode = batch[0][3]

    return KGBatch(
        positive_sample=positive_sample,
        negative_sample=negative_sample,
        subsampling_weight=subsampling_weight,
        mode=mode)


class PytorchKG(Dataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        self.dataset = builder.dataset
        self.split_dict = builder.split_dict
        self.nentity = builder.nentity
        self.nrelation = builder.nrelation

        self.evaluator = Evaluator(name='ogbl-wikikg2')

        if self.mode == 'train':
            self.triples = self.split_dict['train']
            logger.info('#train: %d' % len(self.triples['head']))
            self.build_train_count()
        elif self.mode in ['test', 'valid']:
            if self.mode == 'test':
                self.triples = self.split_dict['test']
                logger.info('#test: %d' % len(self.triples['head']))
            elif self.mode == 'valid':
                self.triples = self.split_dict['valid']
                logger.info('#valid: %d' % len(self.triples['head']))

        logger.info(f'#entity ({self.mode}): {self.nentity}')
        logger.info(f'#relation ({self.mode}): {self.nrelation}')

    def __len__(self):
        if self.mode == "train":
            half_size = min(self.configs.epoch_sampling_size // 2 or float('inf'), len(self.triples['head']))
        else:
            half_size = len(self.triples['head'])
        return int(np.ceil(half_size / self.batch_size)) * self.batch_size * 2

    def load_data(self):
        raise NotImplemented

    def build_train_count(self):
        raise NotImplemented

    def collate_fn(self, batch: List):
        return collate_fn(batch)

    def evaluate(self, y_pred, y_ref, metric: str, output_path: str):
        res_key = dict(mrr='mrr_list', hit1='hits@1_list', hit3='hits@3_list', hit10='hits@10_list')
        y_pred_pos = torch.cat([score[:, 0] for score in y_pred], 0)
        y_pred_neg = torch.cat([score[:, 1:] for score in y_pred], 0)
        logger.info("Evaluating %d samples from %s" % (y_pred_pos.shape[0], self.mode))
        res = self.evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})

        return res[res_key[metric]].mean().cpu().item()

    def shuffle(self):
        if self.mode != 'train':
            return

        size = len(self.triples['head'])
        pos = list(range(size))
        random.shuffle(pos)

        for k in self.triples:
            self.triples[k] = [self.triples[k][pos[i]] for i in range(size)]

        logger.info('Dataset shuffled.')
