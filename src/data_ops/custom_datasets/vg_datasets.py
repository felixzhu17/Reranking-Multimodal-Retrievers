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

from src.data_ops.custom_datasets.base_datasets import BaseDataset, DPRBaseDataset



class VisualGenomeDatasetForDPR(DPRBaseDataset, ModuleParser):
    """
    This is a dataset class for VG dataset used for DPR training
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.passages = dataset_dict['passages']
        if 'images' in dataset_dict.keys():
            self.images = dataset_dict['images']
        """
        Negative samples are randomly sampled from the corpus
        Can choose whether sampling can access the full corpus
        """
        self.n_passages = len(self.passages.id2doc) # number of passages

    def __getitem__(self, idx):
        def negative_sampling(img_id, pos_item_ids, num_samples=1):
            """Generate negative samples for a query. ONLY used in training
            Args:
                user_item (int tensor): user id
                num_samples (int, optional): number of samples. Defaults to 1.
            Returns:
                neg_items: list of negative item ids.
            """
            neg_items = []
            
            while len(neg_items) < num_samples:
                # sample num_samples negative items for the user
                img_id = str(img_id)
                while True:
                    # if self.p is not None:
                    neg_item = np.random.randint(low=0, high=self.n_passages-1, size=1)[0]
                    # else:
                    #     neg_item = np.random.choice(self.n_params.n_items, 1, p=self.p)[0]
                    # print(annotations, neg_item)

                    # neg_passage = self.passages.id2doc[str(neg_item)]
                    VALID = True
                    # Validate if this passage is a negative sample
                    # for answer in answers:
                    #     if answer in neg_passage:
                    #         VALID = False
                    if str(neg_item) in pos_item_ids:
                        VALID = False
                    
                    if VALID == True:
                        break
                neg_items.append(neg_item)
            return neg_items
        
        sample = super().__getitem__(idx)
        item = self.data.data_items[idx]
        # these two belong to a positive sample (in annotations)
        passage_id = random.sample(item.pos_item_ids, k=1)[0]

        passage_content = self.passages.id2doc[passage_id]

        sample.update({
            'passage_id': passage_id,
            'passage_content': passage_content,
            'pos_item_ids': item.pos_item_ids,
            'neg_passage_ids': negative_sampling(item.img_id, item.pos_item_ids, self.config.model_config.num_negative_samples),
        })
        return sample


    def collate_fn(self, batch):
        '''
        when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
        a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''
        batched_data = super().collate_fn(batch)

        #############################
        #  Meta Features
        #############################
        question_ids = [sample.img_id for sample in batch]
        passage_ids = [sample.passage_id for sample in batch]
        pos_item_ids = [sample.pos_item_ids for sample in batch]
        neg_item_ids = [
            str(sample.neg_passage_ids) for sample in batch
        ]

        batched_data.update(EasyDict({
            'passage_ids': passage_ids, # currently used pos item
            'question_ids': question_ids,
            'pos_item_ids': pos_item_ids, # annotated pos items (all)
            'neg_item_ids': neg_item_ids, # currently used neg items
        }))

        return batched_data