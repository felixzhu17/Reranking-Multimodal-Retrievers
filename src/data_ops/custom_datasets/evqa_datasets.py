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

from src.utils.dirs import create_dirs
from src.utils.vqa_tools import VQA
from src.utils.vqaEval import VQAEval
from src.utils.cache_system import save_cached_data, load_cached_data
from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from src.data_ops.custom_datasets.module_parser import ModuleParser

from src.data_ops.custom_datasets.base_datasets import BaseDataset, DPRBaseDataset

class EVQADataset(BaseDataset, ModuleParser):
    """
    Base OKVQA dataset class
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        if 'images' in dataset_dict.keys():
            self.images = dataset_dict['images']
        if 'image_dataset_with_embeddings' in dataset_dict.keys():
            self.image_dataset_with_embeddings = dataset_dict['image_dataset_with_embeddings']
            self.image_dataset_with_embeddings = self.image_dataset_with_embeddings.to_pandas().set_index("__index_level_0__").to_dict(orient="index")
        
        # self.use_ids = []
        # data_items = []
        # for index, item in enumerate(self.data.data_items):
        #     if item['question_id'] in [1552915, 3898695, 4281785, 1631185]:
        #         self.use_ids.append(index)
        #         data_items.append(item)
        # self.data.data_items = data_items


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # to_use_id = random.sample(self.use_ids, 1)[0]
        # sample = super().__getitem__(idx)
        # item = self.data.data_items[idx]
        # sample = EasyDict({
        #     'img_caption': item.img_caption,
        #     'img_ocr': item.img_ocr,
        #     'question_id':  item.question_id,
        #     'question': item.question,
        #     'img_key_full': item.img_key_full,
        #     'img': item.img,
        #     'img_path': item.img_path,
        #     'gold_answer': item.gold_answer,
        #     'answers': item.answers,
        #     'objects': item.objects,
        # })
        sample = EasyDict(self.data[idx])
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
        question_ids = [sample.question_id for sample in batch]
        questions = [sample.question for sample in batch]
        answers = [sample.answers for sample in batch]
        gold_answers = [sample.gold_answer for sample in batch]
        pos_item_ids = [sample.pos_item_ids for sample in batch]
        
        batched_data.update(EasyDict({
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'gold_answers': gold_answers,
            'pos_item_ids': pos_item_ids,
        }))

        return batched_data

