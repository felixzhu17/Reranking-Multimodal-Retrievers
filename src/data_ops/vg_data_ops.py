"""
This file defines the data transforms that will be applied to the data. 
Each transform takes in an EasyDict object of in_features (key: feature_name, value: feature data)
It should output an EasyDict object of out_features (key: feature_name, value: feature_data)
Each transform defined here can be used as an independent unit to form a data pipeline
Some common transforms are provided by runway
"""
from runway_for_ml.data_module.data_transforms import BaseTransform, HFDatasetTransform, register_transform_functor, keep_ds_columns

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
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logger = logging.getLogger(__name__)

from utils.dirs import create_dirs


@register_transform_functor
class LoadVisualGenomeData(BaseTransform):
    """
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, *args, **kwargs):   
        
        module_config = self.module_config

        ######################
        #   Read VG data
        ######################
        
        image_data_path = module_config.data_paths.image_data_path
        image_meta_file = module_config.data_paths.image_meta_file
        region_description_file = module_config.data_paths.region_description_file
        
        with open(image_meta_file, 'r') as f:
            image_meta = json.load(f)
        
        with open(region_description_file, 'r') as f:
            region_descriptions = json.load(f)
        
        all_region_descriptions = {}
        for entry in region_descriptions:
            image_id = entry['id']
            all_region_descriptions[image_id] = entry

        # {'width': 800, 'url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg', 'height': 600, 'image_id': 1, 'coco_id': None, 'flickr_id': None}

        self.data.images = {}
        self.data.vg_data = EasyDict({
            'data_items': [],
        })

        if self.use_dummy_data:
            image_meta = image_meta[:20]

        for img_dict in tqdm(image_meta):
            img_id = img_dict['image_id']
            img_dir = 'VG_100K_2' if 'VG_100K_2' in img_dict['url'] else 'VG_100K'
            img_path = os.path.join(image_data_path, img_dir, f'{img_id}.jpg')

            description_list = all_region_descriptions[img_id]
            
            # img = cv2.imread(img_path)
            # self.data.images[img_path] = img

            entry_data = EasyDict()
            entry_data.img_path = img_path
            entry_data.img_id = img_id
            entry_data.img_meta = img_dict
            entry_data.descriptions = description_list
            
            self.data.vg_data.data_items.append(entry_data)

        logger.info('[Data Statistics] VG data {}'.format(
                        len(self.data.vg_data.data_items)))
        
        return self.data



@register_transform_functor
class PrepareVisualGenomeForRetrieval(BaseTransform):
    """
    This functor conducts the following operations
    1. Gather all passages (descriptions) and deduplicate
    2. Assign passage ids
    3. Assign positive passages to each image
    4. Pack examples, split train and valid
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, **kwargs):
        self.data.update(inputs)

        self.data.passages = {
            'id2doc': {},
            'doc2id': {},
        }

        module_config = self.module_config

        train_valid_ratio = 0.8

        logger.info("Loading passages from VG regional descriptions...")
        data_items = self.data.vg_data.data_items
        random.shuffle(data_items)

        vg_data_for_retrieval = EasyDict({
            'train': {'data_items': []},
            'valid': {'data_items': []},
            'lookup': {},
        })

        for index, item in tqdm(enumerate(data_items)):
            entry_data = EasyDict(item.copy())
            pos_item_ids = []
            pos_item_contents = []
            for description_dict in entry_data['descriptions']['regions']:
                description = description_dict['phrase']
                
                passage_id = self.data.passages['doc2id'].get(description, str(len(self.data.passages['doc2id'])))
                self.data.passages['doc2id'].setdefault(description, passage_id)
                self.data.passages['id2doc'][passage_id] = description
                pos_item_ids.append(passage_id)
                pos_item_contents.append(description)

            entry_data.pos_item_ids = pos_item_ids
            entry_data.pos_item_contents = pos_item_contents

            if (index / len(data_items)) >= train_valid_ratio:
                split = 'valid'
            else:
                split = 'train'
            vg_data_for_retrieval[split].data_items.append(entry_data)
            vg_data_for_retrieval['lookup'][str(entry_data.img_id)] = entry_data
        
        self.data.vg_data = EasyDict(vg_data_for_retrieval)

        logger.info('[Data Statistics] passages {}'.format(
                        len(self.data.passages['id2doc'])))
        logger.info('[Data Statistics] VG data train entries {}'.format(
                        len(self.data.vg_data.train['data_items'])))
        logger.info('[Data Statistics] VG data valid entries {}'.format(
                        len(self.data.vg_data.valid['data_items'])))

        return self.data