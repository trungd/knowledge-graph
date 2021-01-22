from dlex import Params, logger
from dlex.datasets import DatasetBuilder
from dlex.datasets.builder import ModelStringOutput
from dlex.torch.utils.ops_utils import maybe_cuda
from .ogbl import PytorchKG
from ogb.linkproppred import LinkPropPredDataset, Evaluator
import torch
from tqdm import tqdm
from collections import defaultdict


class OgblWikiKG2(DatasetBuilder):
    def __init__(self, params: Params):
        super().__init__(params, pytorch_cls=PytorchOgblWikiKG2)
        self.dataset = self.load_dataset()

    @property
    def split_dict(self):
        return self.dataset.get_edge_split()

    def load_dataset(self):
        logger.info("Loading ogbl-wikikg2...")
        return LinkPropPredDataset(name='ogbl-wikikg2', root=self.get_raw_data_dir())

    @property
    def nentity(self):
        return self.dataset.graph['num_nodes']

    @property
    def nrelation(self):
        return int(max(self.dataset.graph['edge_reltype'])[0]) + 1

    def format_output(self, y_pred, batch_item) -> ModelStringOutput:
        return ModelStringOutput("", "", "")


class PytorchOgblWikiKG2(PytorchKG):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    def build_train_count(self):
        train_count = defaultdict(lambda: 4)
        for i in tqdm(range(len(self.triples['head']))):
            head, relation, tail = self.triples['head'][i], self.triples['relation'][i], self.triples['tail'][i]
            train_count[(head, relation)] += 1
            train_count[(tail, -relation - 1)] += 1
        self.count = train_count

    def __getitem__(self, idx):
        is_head_batch = (idx // self.batch_size) % 2 == 0
        idx = (idx // (2 * self.batch_size) * self.batch_size) + idx % self.batch_size

        if idx >= len(self.triples['head']):
            return None

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
        elif self.mode in ["test", "valid"]:
            subsampling_weight = None
            if not self.configs.test_random_sampling:
                if is_head_batch:
                    negative_sample = torch.cat(
                        [torch.LongTensor([head]), torch.from_numpy(self.triples['head_neg'][idx])])
                else:
                    negative_sample = torch.cat(
                        [torch.LongTensor([tail]), torch.from_numpy(self.triples['tail_neg'][idx])])
            else:
                if is_head_batch:
                    negative_sample = torch.cat(
                        [torch.LongTensor([head]), torch.randint(0, self.nentity, size=(self.configs.eval_negative_sample_size,))])
                else:
                    negative_sample = torch.cat(
                        [torch.LongTensor([tail]), torch.randint(0, self.nentity, size=(self.configs.eval_negative_sample_size,))])

        return maybe_cuda(positive_sample), maybe_cuda(negative_sample), maybe_cuda(subsampling_weight), 'head-batch' if is_head_batch else 'tail-batch'