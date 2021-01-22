from random import random

from dlex import Params, logger
from dlex.datasets import DatasetBuilder
from dlex.datasets.builder import ModelStringOutput
from dlex.torch.utils.ops_utils import maybe_cuda
from .ogbl import PytorchKG
from ogb.linkproppred import LinkPropPredDataset, Evaluator
import torch
from tqdm import tqdm
from collections import defaultdict


class OgblBioKG(DatasetBuilder):
    def __init__(self, params):
        super().__init__(params, pytorch_cls=PytorchOgblBioKG)

        self.dataset = self.load_dataset()
        self.split_dict = self.dataset.get_edge_split()

        entity_dict = dict()
        cur_idx = 0
        for key in self.dataset[0]['num_nodes_dict']:
            entity_dict[key] = (cur_idx, cur_idx + self.dataset[0]['num_nodes_dict'][key])
            cur_idx += self.dataset[0]['num_nodes_dict'][key]
        self.entity_dict = entity_dict

    def load_dataset(self):
        logger.info("Loading ogbl-biokg...")
        return LinkPropPredDataset(name='ogbl-biokg', root=self.get_raw_data_dir())

    @property
    def nentity(self):
        return sum(self.dataset[0]['num_nodes_dict'].values())

    @property
    def nrelation(self):
        return int(max(self.split_dict['train']['relation'])) + 1

    def format_output(self, y_pred, batch_item) -> ModelStringOutput:
        return ModelStringOutput("", "", "")


class PytorchOgblBioKG(PytorchKG):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        self.entity_dict = builder.entity_dict

    def build_train_count(self):
        train_count = defaultdict(lambda: 4)
        for i in tqdm(range(len(self.triples['head']))):
            head, relation, tail = self.triples['head'][i], self.triples['relation'][i], self.triples['tail'][i]
            head_type, tail_type = self.triples['head_type'][i], self.triples['tail_type'][i]
            train_count[(head, relation, head_type)] += 1
            train_count[(tail, -relation - 1, tail_type)] += 1
        self.count = train_count

    def __getitem__(self, idx):
        is_head_batch = (idx // self.batch_size) % 2 == 0
        idx = (idx // (2 * self.batch_size) * self.batch_size) + idx % self.batch_size

        if idx >= len(self.triples['head']):
            return None

        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]

        positive_sample = torch.LongTensor(
            (head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]))

        if self.mode == "train":
            subsampling_weight = self.count[(head, relation, head_type)] + self.count[(tail, -relation - 1, tail_type)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

            if is_head_batch:
                negative_sample = torch.randint(
                    self.entity_dict[head_type][0], self.entity_dict[head_type][1],
                    (self.configs.train_negative_sample_size,))
            else:
                negative_sample = torch.randint(
                    self.entity_dict[tail_type][0], self.entity_dict[tail_type][1],
                    (self.configs.train_negative_sample_size,))
        elif self.mode in ["test", "valid"]:
            subsampling_weight = None
            if not self.configs.test_random_sampling:
                if is_head_batch:
                    negative_sample = torch.cat([
                        torch.LongTensor([head + self.entity_dict[head_type][0]]),
                        torch.from_numpy(self.triples['head_neg'][idx] + self.entity_dict[head_type][0])])
                else:
                    negative_sample = torch.cat([
                        torch.LongTensor([tail + self.entity_dict[tail_type][0]]),
                        torch.from_numpy(self.triples['tail_neg'][idx] + self.entity_dict[tail_type][0])])
            else:
                if is_head_batch:
                    negative_sample = torch.cat([
                        torch.LongTensor([head + self.entity_dict[head_type][0]]),
                        torch.randint(self.entity_dict[head_type][0], self.entity_dict[head_type][1], size=(self.configs.eval_negative_sample_size,))])
                else:
                    negative_sample = torch.cat([
                        torch.LongTensor([tail + self.entity_dict[tail_type][0]]),
                        torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], size=(self.configs.eval_negative_sample_size,))])

        return maybe_cuda(positive_sample), maybe_cuda(negative_sample), maybe_cuda(subsampling_weight), 'head-batch' if is_head_batch else 'tail-batch'