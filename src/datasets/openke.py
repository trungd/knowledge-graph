from typing import List

from dlex import logger
from dlex.datasets import DatasetBuilder
from dlex.datasets.builder import ModelStringOutput
from dlex.datasets.torch import Dataset
from .ogbl import collate_fn

import torch
from ogb.linkproppred import Evaluator


class OpenKE(DatasetBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_output(self, y_pred, batch_item) -> ModelStringOutput:
        return ModelStringOutput("", "", "")


class PytorchOpenKE(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = Evaluator(name='ogbl-wikikg2')

    def collate_fn(self, batch: List):
        return collate_fn(batch)

    def evaluate(self, y_pred, y_ref, metric: str, output_path: str):
        res_key = dict(mrr='mrr_list', hit1='hits@1_list', hit3='hits@3_list', hit10='hits@10_list')
        y_pred_pos = torch.cat([score[:, 0] for score in y_pred], 0)
        y_pred_neg = torch.cat([score[:, 1:] for score in y_pred], 0)
        logger.info("Evaluating %d samples from %s" % (y_pred_pos.shape[0], self.mode))
        res = self.evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})

        return res[res_key[metric]].mean().cpu().item()