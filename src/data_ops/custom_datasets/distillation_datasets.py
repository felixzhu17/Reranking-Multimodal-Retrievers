import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import cv2
import base64

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging

logger = logging.getLogger(__name__)


from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from src.data_ops.custom_datasets.module_parser import ModuleParser

from .dpr_datasets import CommonDatasetForDPR


class CommonDatasetForDistillation(CommonDatasetForDPR):
    """
    This is a dataset class for distillation training
    Scores will be provided.
    """

    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        # self.passages = dataset_dict['passages']
        # if 'images' in dataset_dict.keys():
        #     self.images = dataset_dict['images']
        # if 'image_dataset_with_embeddings' in dataset_dict.keys():
        #     self.image_dataset_with_embeddings = dataset_dict['image_dataset_with_embeddings']
        #     self.image_dataset_with_embeddings = self.image_dataset_with_embeddings.to_pandas().set_index("id").to_dict(orient="index")

        # s = time.time()

        # self.passages = EasyDict({
        #     'dataset': self.passages,
        # })
        # logger.info(f"passages prepared. used {time.time()-s} secs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = EasyDict(self.data[idx])
        item = sample
        # these two belong to a positive sample (in annotations)

        selected_pos_index = 0
        passage_id = item.pos_item_ids[selected_pos_index]
        passage_content = item.pos_item_contents[
            selected_pos_index
        ]  # self.passages.id2doc[passage_id]

        # obtain negative items from the data directly
        neg_passage_ids = item.neg_item_ids[
            : self.config.model_config.num_negative_samples
        ]
        # neg_passage_contents = [
        #     self.passages.dataset[neg_item_index]['passage_content']
        #     for neg_item_index in item.neg_item_indices
        # ]
        neg_passage_contents = item.neg_item_contents[
            : self.config.model_config.num_negative_samples
        ]

        scores = item.scores[: self.config.model_config.num_negative_samples + 1]

        sample.update(
            {
                "img_path": sample["img_path"],
                "passage_id": passage_id,
                "passage_content": passage_content,
                "pos_item_ids": item.pos_item_ids,
                "neg_passage_ids": neg_passage_ids,
                "neg_passage_contents": neg_passage_contents,
                "scores": scores,
            }
        )
        return EasyDict(sample)

    def collate_fn(self, batch):
        """
        when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
        a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        """
        batched_data = super().collate_fn(batch)

        #############################
        #  Meta Features
        #############################
        scores = [sample.scores for sample in batch]
        scores = torch.tensor(scores, dtype=torch.float32)

        batched_data.update(
            {
                "scores": scores,
            }
        )

        return batched_data
