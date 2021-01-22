import os
import random
import shutil

from dlex import Params, logger
from dlex.datasets import DatasetBuilder
from dlex.datasets.builder import ModelStringOutput
from dlex.torch.utils.ops_utils import maybe_cuda
from .ogbl import PytorchKG
from ogb.linkproppred import LinkPropPredDataset, Evaluator
import torch
from tqdm import tqdm

from .wikikg2 import OgblWikiKG2


class OgblResampledWikiKG2(OgblWikiKG2):
    split = 'byrel'

    def __init__(self, params: Params):
        super().__init__(params)
        self._dataset = None

    def load_dataset(self):
        return LinkPropPredDataset(name='ogbl-wikikg2', root=self.get_raw_data_dir())

    @property
    def split_dict(self):
        return self.dataset.get_edge_split(split_type=self.split)

    def maybe_preprocess(self, force=False) -> bool:
        data_root = os.path.join(self.get_raw_data_dir(), 'ogbl_wikikg2')
        if os.path.exists(self.get_processed_data_dir()):
            return False
        os.makedirs(os.path.join(self.get_processed_data_dir(), 'split'), exist_ok=True)

        fract = 0.5  # fraction to take which are tails of the relation
        extra = 1.1  # extra sample so that we can exclude triples in the graph

        train = torch.load(os.path.join(data_root, 'split', 'time', 'train.pt'))
        valid = torch.load(os.path.join(data_root, 'split', 'time', 'valid.pt'))
        test = torch.load(os.path.join(data_root, 'split', 'time', 'test.pt'))

        heads = dict()
        tails = dict()
        ht = {'head': heads, 'tail': tails}
        at = {'head': train['head'].tolist(), 'tail': train['tail'].tolist()}
        all = set()

        # return N samples of field f for relation r
        def sample(N, f, relation, v, ex=extra):
            Nr = int(N * fract)
            Nt = int(N * ex)
            if Nr < len(ht[f][relation]):
                sl = random.sample(ht[f][relation], Nr)
            else:
                sl = ht[f][r]
            sl = sl + random.sample(at[f], Nt - len(sl))
            # remove those for which (x,r,v) or (v,r,x) is in the graph
            if f == 'head':
                sl = [x for x in sl if (x, relation, v) not in all]
            else:
                sl = [x for x in sl if (v, relation, x) not in all]
            # remove duplicates
            sl = list(set(sl))
            if len(sl) < N:
                # print('error: too few negs', len(sl), 'for item', f, relation, v, 'try again')
                sl = sample(N, f, relation, v, ex * ex * N / (len(sl) + 1))
            return sl[:N]

        def triples(l):
            for t in l:
                for i in range(t['head'].shape[0]):
                    yield t['head'][i], t['relation'][i], t['tail'][i]

        for h, r, t in triples([train, valid, test]):
            heads.setdefault(r, []).append(h)
            tails.setdefault(r, []).append(t)
            all.add((h, r, t))

        # also valid
        for i in tqdm(range(test['head'].shape[0])):
            r = test['relation'][i]
            nh, nt = len(heads.setdefault(r, [])), len(tails.setdefault(r, []))
            # if i % 100 == 1:
            #     print(i, r, nh, nt)
            hnl, tnl = len(test['head_neg'][i]), len(test['tail_neg'][i])
            test['head_neg'][i] = sample(hnl, 'head', test['relation'][i], test['tail'][i])
            test['tail_neg'][i] = sample(tnl, 'tail', test['relation'][i], test['head'][i])

        os.makedirs(os.path.join(data_root, 'split', self.split), exist_ok=True)
        shutil.copyfile(os.path.join(data_root, 'split', 'time', 'train.pt'), os.path.join(data_root, 'split', self.split, 'train.pt'))
        shutil.copyfile(os.path.join(data_root, 'split', 'time', 'valid.pt'), os.path.join(data_root, 'split', self.split, 'valid.pt'))

        torch.save(test, os.path.join(data_root, 'split', self.split, 'test.pt'))