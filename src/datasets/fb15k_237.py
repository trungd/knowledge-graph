import os
from collections import defaultdict

import torch

from dlex import logger, List
from dlex.datasets import DatasetBuilder
from dlex.datasets.torch import Dataset

import numpy as np

from torchbiggraph.converters.utils import download_url, extract_tar
from torchbiggraph.config import ConfigFileLoader

from dlex.torch.utils.ops_utils import maybe_cuda
from .openke import PytorchOpenKE, OpenKE


class FB15K237(OpenKE):
    def __init__(self, params):
        super().__init__(
            params,
            downloads=[
                "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/train2id.txt",
                "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/test2id.txt",
                "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/valid2id.txt",
                "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/relation2id.txt",
                "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/entity2id.txt",
            ],
            pytorch_cls=PytorchFB15K237)

    def maybe_preprocess(self, force=False) -> bool:
        pass


class PytorchFB15K237(PytorchOpenKE):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        self.nentity = int(open(os.path.join(builder.get_working_dir(), 'entity2id.txt')).readline())
        self.nrelation = int(open(os.path.join(builder.get_working_dir(), 'relation2id.txt')).readline())

        if self.mode == 'train':
            train_count = defaultdict(lambda: 4)
            for i in range(len(self.data)):
                head, tail, relation = self.data[i]
                train_count[(head, relation)] += 1
                train_count[(tail, -relation - 1)] += 1
            self.count = train_count

    def load_data(self):
        with open(os.path.join(self.builder.get_working_dir(), f'{self.mode}2id.txt')) as f:
            lines = f.read().split('\n')[1:]

        return [[int(x) for x in l.split(' ')] for l in lines if l]

    def __len__(self):
        if self.mode == "train":
            half_size = min(self.configs.epoch_sampling_size // 2 or float('inf'), len(self.data))
        else:
            half_size = len(self.data)
        return int(np.ceil(half_size / self.batch_size)) * self.batch_size * 2

    def __getitem__(self, idx):
        is_head_batch = (idx // self.batch_size) % 2 == 0
        idx = (idx // (2 * self.batch_size) * self.batch_size) + idx % self.batch_size

        if idx >= len(self.data):
            return None

        head, tail, relation = self.data[idx]

        positive_sample = torch.LongTensor((head, relation, tail))

        if self.mode == "train":
            subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

            negative_sample = torch.randint(0, self.nentity, (self.configs.train_negative_sample_size,))
        elif self.mode in ["test", "valid"]:
            subsampling_weight = None
            if is_head_batch:
                negative_sample = torch.cat(
                    [torch.LongTensor([head]), torch.randint(0, self.nentity, size=(self.configs.eval_negative_sample_size,))])
            else:
                negative_sample = torch.cat(
                    [torch.LongTensor([tail]), torch.randint(0, self.nentity, size=(self.configs.eval_negative_sample_size,))])

        return maybe_cuda(positive_sample), maybe_cuda(negative_sample), maybe_cuda(subsampling_weight), 'head-batch' if is_head_batch else 'tail-batch'