from collections import defaultdict
from typing import List

import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import LinkPropPredDataset, Evaluator
from tqdm import tqdm

from dlex import Params, logger
from dlex.datasets import DatasetBuilder
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


class OgblWikiKG2(DatasetBuilder):
    def __init__(self, params: Params):
        super().__init__(params, pytorch_cls=PytorchOgblWikiKG2)


class OgblBioKG(DatasetBuilder):
    def __init__(self, params):
        super().__init__(params, pytorch_cls=PytorchOgblBioKG)


class OgblCitation2(DatasetBuilder):
    def __init__(self, params):
        super().__init__(params, pytorch_cls=PytorchOgblCitation2)


class PytorchKG(Dataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        self.dataset = self.load_dataset()

        split_dict = self.split_dict = self.dataset.get_edge_split()

        self.evaluator = Evaluator(name='ogbl-wikikg2')

        if self.mode == 'train':
            self.triples = split_dict['train']
            logger.info('#train: %d' % len(self.triples['head']))
            train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
            for i in tqdm(range(len(self.triples['head']))):
                head, relation, tail = self.triples['head'][i], self.triples['relation'][i], self.triples['tail'][i]
                train_count[(head, relation)] += 1
                train_count[(tail, -relation - 1)] += 1
                train_true_head[(relation, tail)].append(head)
                train_true_tail[(head, relation)].append(tail)
            self.count = train_count
        elif self.mode == 'test' or self.mode == 'valid':
            if self.mode == 'test':
                self.triples = split_dict['test']
                logger.info('#test: %d' % len(self.triples['head']))
            elif self.mode == 'valid':
                self.triples = split_dict['valid']
                logging.info('#valid: %d' % len(self.triples['head']))

        logger.info(f'#entity ({self.mode}): {self.nentity}')
        logger.info(f'#relation ({self.mode}): {self.nrelation}')

    def load_dataset(self):
        pass

    @property
    def nentity(self):
        return None

    @property
    def nrelation(self):
        return None

    def __len__(self):
        # return 500000
        return 2 * len(self.triples['head'])

    def load_data(self):
        pass

    def __getitem__(self, idx):
        mode = 'head-batch' if idx % 2 == 0 else 'tail-batch'
        idx = idx // 2

        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        positive_sample = torch.LongTensor((head, relation, tail))

        if self.mode == "train":
            subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

            # negative_sample_list = []
            # negative_sample_size = 0

            # while negative_sample_size < self.configs.negative_sample_size:
            #     negative_sample = np.random.randint(self.nentity, size=self.configs.negative_sample_size*2)
            #     if self.mode == 'head-batch':
            #         mask = np.in1d(
            #             negative_sample,
            #             self.true_head[(relation, tail)],
            #             assume_unique=True,
            #             invert=True
            #         )
            #     elif self.mode == 'tail-batch':
            #         mask = np.in1d(
            #             negative_sample,
            #             self.true_tail[(head, relation)],
            #             assume_unique=True,
            #             invert=True
            #         )
            #     else:
            #         raise ValueError('Training batch mode %s not supported' % self.mode)
            #     negative_sample = negative_sample[mask]
            #     negative_sample_list.append(negative_sample)
            #     negative_sample_size += negative_sample.size

            # negative_sample = np.concatenate(negative_sample_list)[:self.configs.negative_sample_size]

            # negative_sample = torch.from_numpy(negative_sample)
            # negative_sample = torch.from_numpy(np.random.randint(self.nentity, size=self.configs.negative_sample_size))

            negative_sample = torch.randint(0, self.nentity, (self.configs.negative_sample_size,))
        elif self.mode == "test":
            subsampling_weight = None
            if mode == 'head-batch':
                if not self.configs.test_random_sampling:
                    negative_sample = torch.cat(
                        [torch.LongTensor([head]), torch.from_numpy(self.triples['head_neg'][idx])])
                else:
                    negative_sample = torch.cat(
                        [torch.LongTensor([head]), torch.randint(0, self.nentity, size=(self.neg_size,))])
            elif mode == 'tail-batch':
                if not self.configs.test_random_sampling:
                    negative_sample = torch.cat(
                        [torch.LongTensor([tail]), torch.from_numpy(self.triples['tail_neg'][idx])])
                else:
                    negative_sample = torch.cat(
                        [torch.LongTensor([tail]), torch.randint(0, self.nentity, size=(self.neg_size,))])

        return maybe_cuda(positive_sample), maybe_cuda(negative_sample), maybe_cuda(subsampling_weight), mode

    def collate_fn(self, batch: List):
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

    def evaluate(self, y_pred, y_ref, metric: str, output_path: str):
        for score in y_pred:
            res = self.evaluator.eval({'y_pred_pos': score[:, 0], 'y_pred_neg': score[:, 1:]})
            if metric == 'mrr':
                return res['mrr_list'].mean().cpu().item()


class PytorchOgblWikiKG2(PytorchKG):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    def load_dataset(self):
        logger.info("Loading ogbl-wikikg2...")
        return LinkPropPredDataset(name='ogbl-wikikg2', root=self.builder.get_raw_data_dir())

    @property
    def nentity(self):
        return self.dataset.graph['num_nodes']

    @property
    def nrelation(self):
        return int(max(self.dataset.graph['edge_reltype'])[0]) + 1


class PytorchOgblBioKG(PytorchKG):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    def load_dataset(self):
        logger.info("Loading ogbl-biokg...")
        return LinkPropPredDataset(name='ogbl-biokg', root=self.builder.get_raw_data_dir())

    @property
    def nentity(self):
        return sum(self.dataset[0]['num_nodes_dict'].values())

    @property
    def nrelation(self):
        return int(max(self.triples['relation'])) + 1


class PytorchOgblCitation2(PytorchKG):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    def load_dataset(self):
        logger.info("Loading ogbl-citation2...")
        return LinkPropPredDataset(name='ogbl-citation2', root=self.builder.get_raw_data_dir())

    @property
    def nentity(self):
        return sum(self.dataset[0]['num_nodes_dict'].values())

    @property
    def nrelation(self):
        return int(max(self.triples['relation'])) + 1
