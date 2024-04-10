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

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from utils.cache_system import save_cached_data, load_cached_data
from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from data_ops.custom_datasets.module_parser import ModuleParser

from .base_datasets import BaseDataset, DPRBaseDataset

class InfoseekDataset(BaseDataset, ModuleParser):
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
        wikidata_ranges = [sample.wikidata_range for sample in batch]
        wikidata_values = [sample.wikidata_value for sample in batch]
        pos_item_ids = [sample.pos_item_ids for sample in batch]
        
        batched_data.update(EasyDict({
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'gold_answers': gold_answers,
            'wikidata_ranges': wikidata_ranges,
            'wikidata_values': wikidata_values,
            'pos_item_ids': pos_item_ids,
        }))

        return batched_data



class InfoseekDatasetForDPR(DPRBaseDataset, ModuleParser):
    """
    This is a dataset class for OKVQA dataset used for DPR training
    """
    def __init__(self, config, dataset_dict):
        super().__init__(config, dataset_dict)
        self.cutoff = dataset_dict.get('cutoff', None)
        if self.cutoff is not None:
            cutoff = min(self.cutoff, len(self.data))
            self.data = self.data.select(range(cutoff))
        
        self.question_prefix = dataset_dict.get('question_prefix', None)
        
        self.passages = dataset_dict['passages']
        self.filtered_passages = dataset_dict['filtered_passages']
        if 'images' in dataset_dict.keys():
            self.images = dataset_dict['images']
        if 'image_dataset_with_embeddings' in dataset_dict.keys():
            self.image_dataset_with_embeddings = dataset_dict['image_dataset_with_embeddings']
            self.image_dataset_with_embeddings = self.image_dataset_with_embeddings.to_pandas().set_index("__index_level_0__").to_dict(orient="index")

        s = time.time()
        # if self.config.model_config.full_corpus_in_testing or self.config.model_config.full_corpus_in_training:
        #     load_full_corpus = True
        # else:
        #     load_full_corpus = False
        
        if self.config.model_config.full_corpus_in_training:
            self.passages = EasyDict({
                'dataset': self.passages,
                'id2doc': {},
                'id2doc_train': {},
            })
        else:
            self.passages = EasyDict({
                'dataset': self.filtered_passages,
                'id2doc': {},
                'id2doc_train': {},
            })
        
        # # first, load filtered passage as train docs
        # ds = self.filtered_passages
        # logger.info(f"Using {len(ds)} passage data...")
        # self.passages.id2doc_train = ds.to_pandas().set_index("id").to_dict(orient="index")
        # for k, v in tqdm(self.passages.id2doc_train.items(), desc="formatting the passages"):
        #     self.passages.id2doc_train[k] = f"title: {v['title']} content: {v['text']}"
        
        # if load_full_corpus:
        #     # Load full corpus only when needed
        #     ds = self.passages.dataset
        #     logger.info(f"Using {len(ds)} passage data...")
        #     self.passages.id2doc = ds.to_pandas().set_index("id").to_dict(orient="index")
        #     for k, v in tqdm(self.passages.id2doc.items(), desc="formatting the passages"):
        #         self.passages.id2doc[k] = f"title: {v['title']} content: {v['text']}"
        # else:
        #     self.passages.id2doc = self.passages.id2doc_train
        
        logger.info(f"passages prepared. used {time.time()-s} secs.")
        
        """
        Negative samples are randomly sampled from the corpus
        Can choose whether sampling can access the full corpus
        """
        # if self.mode == 'train':
        #     if not self.config.model_config.full_corpus_in_training:
        #         # random sampling for training is limited to a small fraction
        #         self.n_passages = len(self.passages.id2doc_train) # number of passages
        #         self.available_ids = list(self.passages.id2doc_train.keys())
        #     else:
        #         self.n_passages = len(self.passages.id2doc)
        #         self.available_ids = list(self.passages.id2doc.keys())
        # else:
        #     # while testing, negative samples are not used; do not need to change this value
        #     self.n_passages = len(self.passages.id2doc) # number of passages
        #     self.available_ids = list(self.passages.id2doc.keys())
        self.n_passages = len(self.passages.dataset)
        print('available ids', self.n_passages)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        def negative_sampling(question_id, pos_item_ids, num_samples=1):
            """Generate negative samples for a query. ONLY used in training
            Args:
                user_item (int tensor): user id
                num_samples (int, optional): number of samples. Defaults to 1.
            Returns:
                neg_items: list of negative item ids.
            """
            neg_items = []
            #annotations = self.passages.annotations.get(str(question_id), {'passages': []})['passages']
            
            while len(neg_items) < num_samples:
                # sample num_samples negative items for the user
                question_id = str(question_id)
                while True:
                    # if self.p is not None:
                    neg_item = np.random.randint(low=0, high=self.n_passages-1, size=1)[0]
                    # else:
                    #     neg_item = np.random.choice(self.n_params.n_items, 1, p=self.p)[0]
                    # print(annotations, neg_item, type(neg_item))

                    # neg_passage = self.passages.id2doc[str(neg_item)]
                    VALID = True
                    # Validate if this passage is a negative sample
                    # for answer in answers:
                    #     if answer in neg_passage:
                    #         VALID = False
                    neg_item = self.passages.dataset[int(neg_item)]
                    if neg_item['passage_id'] in pos_item_ids:
                        VALID = False
                    
                    if VALID == True:
                        break
                neg_items.append(neg_item)
            return neg_items
        
        sample = EasyDict(self.data[idx])
        item = sample
        # randomly sample one positive sample
        selected_pos_index = random.sample(range(len(item.pos_item_ids)), k=1)[0]
        passage_id = item.pos_item_ids[selected_pos_index]
        passage_content = item.pos_item_contents[selected_pos_index] #self.passages.id2doc[passage_id]
        
        # passage_content = self.passages.id2doc[str(passage_id)]
        
        neg_items = negative_sampling(item.question_id, item.pos_item_ids, self.config.model_config.num_negative_samples)
        neg_passage_ids = [neg_item['passage_id'] for neg_item in neg_items]
        neg_passage_contents = [neg_item['passage_content'] for neg_item in neg_items]

        if self.question_prefix is not None:
            sample.question = random.choice(self.question_prefix) + ' ' + sample.question
        
        sample.update({
            'img_path': sample.image_path,
            'passage_id': passage_id,
            'passage_content': passage_content,
            'pos_item_ids': item.pos_item_ids,
            'neg_passage_ids': neg_passage_ids,
            'neg_passage_contents': neg_passage_contents,
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
        question_ids = [sample.question_id for sample in batch]
        questions = [sample.question for sample in batch]
        answers = [sample.answers for sample in batch]
        gold_answers = [sample.gold_answer for sample in batch]
        passage_ids = [sample.passage_id for sample in batch]
        pos_item_ids = [sample.passage_id for sample in batch]
        neg_item_ids = [
            sample.neg_passage_ids for sample in batch
        ]

        batched_data.update(EasyDict({
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'gold_answers': gold_answers,
            'passage_ids': passage_ids,
            'pos_item_ids': pos_item_ids,
            'neg_item_ids': neg_item_ids,
        }))

        return batched_data



