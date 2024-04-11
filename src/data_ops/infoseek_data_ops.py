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
import math
import psutil

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logger = logging.getLogger(__name__)

from src.utils.dirs import create_dirs
from src.utils.vqa_tools import VQA
from src.utils.vqaEval import VQAEval

from transformers import AutoImageProcessor, CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
from torchvision import transforms
from diffusers import AutoencoderKL
from datasets import Dataset, DatasetDict, concatenate_datasets
import PIL

from src.models.custom_clip_processor import CustomCLIPImageProcessor


@register_transform_functor
class LoadInfoSeekData(HFDatasetTransform):
    """
    """
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        
    def _call(self, inputs, **kwargs):
        for input_data in inputs:
            self.data.update(input_data)

        module_config = self.module_config

        ######################
        #   Read OK-VQA data
        ######################
        def most_frequent(List):
            return max(set(List), key = List.count)
        

        self.data.infoseek_data = EasyDict({
            'lookup': {},
        })
        self.data.images = EasyDict({})
        
        
        imgid2path = {}
        for image_data_path in module_config.image_data_path.values():
            for root, dirs, files in os.walk(image_data_path):
                for file in tqdm(files):
                    if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                        image_path = os.path.join(image_data_path, file)
                        image_id = file.split(".")[0]
                        imgid2path[image_id] = image_path
        


        print(f"Image loaded, total {len(imgid2path)}")

        self.output_data = DatasetDict()

        # Load Infoseek data
        for data_split, jsonl_path in module_config.vqa_data_path.items():
            batch_data = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    line_data = json.loads(line)
                    if self.data.vinvl_features.get(line_data['image_id'], None) is None:
                        continue
                    if line_data.get('answer_eval', None) is not None:
                        # if isinstance(line_data['answer_eval'][0], list):
                        #     line_data['answer_eval'] = {
                        #         "answer_list": line_data['answer_eval']
                        #     }
                        if isinstance(line_data['answer_eval'][0], dict):
                            # if len(line_data['answer_eval']) > 1:
                            #     print(line_data['answer_eval'])
                            line_data['wikidata_value'] = line_data['answer_eval'][0]['wikidata']
                            line_data['wikidata_range'] = line_data['answer_eval'][0]['range']
                            line_data['answer_eval'] = []
                    batch_data.append(line_data)

            logger.info(f"Total {len(batch_data)} examples in {data_split} split")

            if self.use_dummy_data:
                batch_data = batch_data[:20]

            df_split = pd.DataFrame.from_records(batch_data)


            # read withkb data
            annotation_file_path = module_config.passage_annotation_file_path.get(data_split, None)
            if annotation_file_path is not None:
                batch_data = []
                with open(annotation_file_path, 'r') as f:
                    for line in f:
                        line_data = json.loads(line)
                        batch_data.append(line_data)

                # convert to dataframe
                print(df_split.head())
                df_split_kb = pd.DataFrame.from_records(batch_data)
                print(df_split_kb.head())
                # joint ds_split and ds_split_kb based on "data_id"
                df_split = pd.merge(df_split, df_split_kb, on='data_id', how="inner")
                print(df_split.head())

            ds_split = Dataset.from_pandas(df_split)
            ds_split = ds_split.rename_columns({"data_id": "question_id", "answer": "answers"})

            def attach_image_path(example):
                example["image_path"] = imgid2path.get(example["image_id"], "")
                return example

            ds_split = ds_split.map(attach_image_path)
            # filter out examples without image
            logger.info(f"before filter: {len(ds_split)}")
            ds_split = ds_split.filter(lambda example: example["image_path"] != "")
            logger.info(f"after filter: {len(ds_split)}")
            

            def convert_format(example):
                example['gold_answer'] = most_frequent(example['answers'])
                return example

            ds_split = ds_split.map(convert_format, desc="convert_format")

            def add_features(entry_data):
                if module_config.add_VinVL_features:
                    # Read predictions from VinVL features
                    VinVL_prediction = self.data.vinvl_features.get(entry_data['image_id'], None)
                    if not VinVL_prediction:
                        logger.error(f"Image {entry_data['image_id']} does not find associated VinVL features!")
                        # raise KeyError(f"Image {entry_data['image_id']} does not find associated VinVL features!")
                        entry_data['objects'] = []
                    else:
                        objects = []
                        for obj in VinVL_prediction['objects']:
                            # obj_feature = np.frombuffer(base64.b64decode(obj['feature']), np.float32)
                            # obj_feature_ts = torch.FloatTensor(obj_feature.copy())
                            obj_class = obj['class']
                            obj_rect = obj['rect']
                            obj_attributes = obj['attributes']
                            obj_attribute_scores = obj['attr_scores']
                            obj_ocr = obj.get('ocr', [])
                            objects.append({
                                'class': obj_class,
                                # 'obj_feature': obj_feature_ts,
                                'rect': obj_rect,
                                'attributes': obj_attributes,
                                'attribute_scores': obj_attribute_scores,
                                'ocr': obj_ocr,
                            })
                        entry_data['objects'] = objects
                return entry_data
            
            ds_split = ds_split.map(add_features, desc="add_features")
            ds_split = ds_split.filter(lambda example: len(example['objects']) > 0)
            logger.info(f"after adding features: {len(ds_split)}")

            if data_split == 'train':
                num_examples_limit = module_config.num_train
            else:
                num_examples_limit = module_config.num_valid
            
            if num_examples_limit > 0 and len(ds_split) > num_examples_limit:
                ds_split = ds_split.select(range(num_examples_limit))
                logger.info(f"Select {num_examples_limit} examples from {data_split} split")

            for img_path in ds_split['image_path']:
                self.data.images[img_path] = {
                    'img_path': img_path,
                }

            print(ds_split)
            print(ds_split[0])

            # self.data.infoseek_data[data_split] = EasyDict()
            # self.data.infoseek_data[data_split].dataset = ds_split

            self.output_data[data_split] = ds_split
            # for entry_data in self.data.okvqa_data[data_split].data_items:
            #     self.data.okvqa_data['lookup'][str(entry_data.question_id)] = entry_data

            # Report statistics
            logger.info('[Data statistics] split: {}  entries: {}'.format(
                data_split, len(ds_split)))

        df = pd.DataFrame.from_dict(self.data.images, orient="index")
        image_dataset = Dataset.from_pandas(df)

        self.output_data['images'] = image_dataset
        
        # NOTE: we have merged features with items directly, 
        #       so they are not required in later stages
        print(self.output_data)

        return self.output_data




@register_transform_functor
class CropRegionOfInterestImagesForInfoSeek(HFDatasetTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function crops ROIs from the object detection results of each image
        These ROIs are saved to data.images so that their features are processed automatically.
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        self.out_data = DatasetDict()

        images = {}

        for data_split in ['train', 'val']:
            def crop_ROI(item):
                item = EasyDict(item)
                # print(item.question)
                selected_objects = []
                objects = []
                for obj in item.objects:
                    # print(obj['rect'], obj['class'])
                    xmin, ymin, xmax, ymax = obj['rect']
                    obj_area = (ymax - ymin) * (xmax - xmin)
                    objects.append((obj_area, obj))
                    if obj['class'].lower().strip() in item.question.lower():
                        selected_objects.append(obj)
                
                objects = sorted(objects, key=lambda x: x[0], reverse=True)
                
                for obj_area, obj in objects:
                    xmin, ymin, xmax, ymax = obj['rect']
                    if len(selected_objects) >= self.module_config.max_objects:
                        break
                    else:
                        valid = True
                        # Remove duplications
                        for existing_obj in selected_objects:
                            if existing_obj['class'] == obj['class']:
                                e_xmin, e_ymin, e_xmax, e_ymax = existing_obj['rect']
                                if xmin >= e_xmin and ymin >= e_ymin and xmax <= e_xmax and ymax <= e_ymax:
                                    # this object is contained in an existing object with the same class name
                                    valid = False
                        if valid:
                            selected_objects.append(obj)

                img_path = item.image_path

                if len(selected_objects) == 0:
                    selected_objects = []
                else:
                    if len(selected_objects) > self.module_config.max_objects:
                        selected_objects = selected_objects[:min(self.module_config.max_objects, len(selected_objects))]

                ROIs = []
                for obj in selected_objects:
                    xmin, ymin, xmax, ymax = obj['rect']
                    xmin, ymin, xmax, ymax = round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)
                    new_id = f"{img_path}|||{obj['class']}_{xmin}_{ymin}_{xmax}_{ymax}"
                    new_img_dict = {
                        'img_path': img_path,
                        'obj': obj,
                        'crop': [xmin, ymin, xmax, ymax],
                    }
                    images[new_id] = new_img_dict
                    ROIs.append(new_id)
                
                item.ROIs = ROIs
                return item

            ds_split = self.data[data_split]
            cropped_ds_split = ds_split.map(crop_ROI, desc="crop_ROI", load_from_cache_file=False)
            self.out_data[data_split] = cropped_ds_split

        additional_image_df = pd.DataFrame.from_dict(images, orient="index")
        additional_image_dataset = Dataset.from_pandas(additional_image_df)

        # concatenate with existing image dataset
        self.out_data['images'] = concatenate_datasets([self.data['images'], additional_image_dataset])

        print(self.out_data)

        return self.out_data
    



@register_transform_functor
class PrepareWikipediaPassageAnnotationsForInfoSeek(HFDatasetTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function prepares Wikipedia passage annotations
        The annotation can be pseudo labels or ground-truth labels
        {
            index_name: "wikipedia",
            supervision_type: "pseudo" or "ground-truth",
        },
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        module_config = self.module_config

        # Prepare ElasticSearch
        from elasticsearch import Elasticsearch, helpers

        # Password for the 'elastic' user generated by Elasticsearch
        ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]

        es = Elasticsearch(
                "https://localhost:9200",
                ca_certs=os.environ["ELASTIC_CA_CERTS"],
                basic_auth=("elastic", ELASTIC_PASSWORD),
        )

        # Successful response!
        es.info()

        ds = self.data.passages.dataset
        index_name = module_config.index_name

        available_documents = {}
        
        def search_for_a_string(es, query, fileds=["title", "text"], size=10, from_=0):
            resp = es.search(index=index_name, query={
                "multi_match" : {
                    "query": query,
                    "fields": fileds,
                    "type": "phrase",
                }
            },  size=size, from_=from_, timeout="60s")
            # pprint(resp)
            # print("Got %d Hits:" % resp['hits']['total']['value'])
            # for hit in resp['hits']['hits']:
            #     print(hit)
            return resp
        
        # def match_a_question(query):
        #     resp = es.search(index="wikipedia", query={
        #         "match" : {
        #             "text": {
        #                 "query": query,
        #                 "fuzziness": "AUTO"
        #             }
        #         }
        #     })
        #     pprint(resp)
        #     return resp

        output_data = DatasetDict()

        if module_config.supervision_type == "ground-truth":
            self.data.infoseek_data_with_dpr_output = EasyDict()

            ######################
            #  Get ground-truth passages
            ######################

            # Read associated title from the dataset
            for data_split in ['train', 'val']:
                ds_split = self.data[data_split]

                def search_wiki_passage_with_entity_text(example):
                    es = Elasticsearch(
                        "https://localhost:9200",
                        ca_certs=os.environ["ELASTIC_CA_CERTS"],
                        basic_auth=("elastic", ELASTIC_PASSWORD),
                        timeout=60,
                    )
                    example = EasyDict(example)
                    query = example.entity_text
                    # print(f"Searching for {example.question_id} {example.question}: {query}  answers: {example.answers}")
                    resp = search_for_a_string(es, query, fileds=["title"], size=1000)

                    gold_doc_ids = []
                    gold_doc_contents = []
                    related_doc_ids = []

                    all_answers = example.answers
                    all_answers += example.answer_eval

                    # else:
                    #     all_answers += example.wikidata_value
                    #     all_answers += example.wikidata_range[0]
                    #     all_answers += example.wikidata_range[1]
                    # print("all_answers", all_answers)


                    if resp['hits']['total']['value'] > 0:
                        doc_title = resp['hits']['hits'][0]['_source']['title']

                        for retrieved_doc in resp['hits']['hits']:
                            if retrieved_doc['_source']['title'] != doc_title:
                                # found = False
                                # for answer in all_answers:
                                #     if answer.lower() in passage_text.lower():
                                #         # print("!! Found answer", answer)
                                #         found = True
                                #         break
                                # if not found:
                                #     related_doc_ids.append(retrieved_doc['_id'])
                                continue
                            passage_text = retrieved_doc['_source']['text']
                            
                            # pprint(retrieved_doc)
                            found = False
                            for answer in all_answers:
                                if answer.lower() in passage_text.lower():
                                    # print("!! Found answer", answer)
                                    found = True
                                    gold_doc_ids.append(retrieved_doc['_id'])
                                    gold_doc_contents.append(passage_text)
                                    break
                            
                            if not found and example.wikidata_value is not None:
                                # wikidata range
                                # find all float and integer values in passage_text using regular expression, the value can be comma separated
                                all_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", passage_text)
                                
                                for number in all_numbers:
                                    try:
                                        number = float(number)
                                        if abs(number-example.wikidata_value) < 0.01: # >= float(example.wikidata_range[0]) and number <= float(example.wikidata_range[1]):
                                            print("!! Found answer", number, '==', example.wikidata_value)
                                            found = True
                                            gold_doc_ids.append(retrieved_doc['_id'])
                                            gold_doc_contents.append(passage_text)
                                            break
                                    except:
                                        continue
                                
                            related_doc_ids.append(retrieved_doc['_id'])
                            
                        # passage_text = resp['hits']['hits'][0]['_source']['text']
                        # passage_title = resp['hits']['hits'][0]['_source']['title']
                        # example.passages = [i['_id'] for i in resp['hits']['hits']]
                        # print("related_doc_ids", related_doc_ids)
                        # print("gold_doc_ids", gold_doc_ids)
                        example.related_item_ids = related_doc_ids
                        example.pos_item_ids = gold_doc_ids
                        example.pos_item_contents = gold_doc_contents

                        # for doc_id in related_doc_ids:
                        #     available_documents[doc_id] = 1
                    else:
                        example.related_item_ids = []
                        example.pos_item_ids = []
                        example.pos_item_contents = []
                        logger.error(f"Cannot find a passage for {example.question_id}: {query}")
                    
                    return example
                
                ds_split = ds_split.map(search_wiki_passage_with_entity_text, desc="search_wiki_passage_with_entity_text", load_from_cache_file=False, num_proc=128)
                # for example in tqdm(ds_split):
                #     for doc_id in example.related_doc_ids:
                #         available_documents[doc_id] = 1

                # remove examples without passages
                logger.info(f"Before removing examples without passages: {len(ds_split)}")
                ds_split = ds_split.filter(lambda x: len(x['pos_item_ids']) != 0, desc="remove_examples_without_pos_item_ids", num_proc=128)
                logger.info(f"After removing examples without passages: {len(ds_split)}")

                if data_split == 'train':
                    num_examples_limit = module_config.num_train
                else:
                    num_examples_limit = module_config.num_valid
                
                if num_examples_limit > 0 and len(ds_split) > num_examples_limit:
                    ds_split = ds_split.select(range(num_examples_limit))
                    logger.info(f"Select {num_examples_limit} examples from {data_split} split")

                pprint(ds_split[0])

                # Report statistics
                logger.info('[Data statistics] loaded with knowledge data split: {}  entries: {}'.format(
                    data_split,
                    len(ds_split)))
                
                output_data[data_split] = ds_split

        print(f"total #docs {len(ds)}")
        # print(f"total #docs with answers {len(available_documents)}")

        return output_data
    


@register_transform_functor
class ReduceWikipediaPassagesSizeForInfoSeek(HFDatasetTransform):
    def setup(self, num_random_passages=0, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config
        self.num_random_passages = num_random_passages

    def _call(self, inputs, *args, **kwargs):
        """
        This function reduces passages size for training and testing
        {
        },
        """
        for input_data in inputs:
            self.data.update(input_data)
        
        module_config = self.module_config

        full_ds = self.data.passages.dataset

        # get all available documents from related_item_ids
        available_documents = {}
        for data_split in ['train', 'val']:
            ds_split = self.data[data_split]
            all_related_item_ids = ds_split['related_item_ids']
            for related_item_ids in all_related_item_ids:
                for doc_id in related_item_ids:
                    available_documents[doc_id] = 1

        print(f"total #docs {len(available_documents)}")
        
        # Randomly select self.num_random_passages passages
        if self.num_random_passages > 0:
            random_passages = random.sample(list(full_ds['id']), self.num_random_passages)
            for doc_id in random_passages:
                available_documents[doc_id] = 1
        
        # filter out docs that don't have answers
        filtered_ds = full_ds.filter(lambda x: available_documents.get(x['id'], None) is not None, load_from_cache_file=False)
        
        # rename id to passage_id, and text to passage_content
        filtered_ds = filtered_ds.rename_columns({"id": "passage_id", "text": "passage_content"})
        full_ds = full_ds.rename_columns({"id": "passage_id", "text": "passage_content"})

        output_data = DatasetDict({
            "filtered_passages": filtered_ds,
            "passages": full_ds,
        })
        logger.info(f"reducing #docs {len(full_ds)} (full) to {len(filtered_ds)}")
        return output_data
    



@register_transform_functor
class CaptionImageWithBLIP2(HFDatasetTransform):
    def setup(self, pretrained_model_name='Salesforce/blip2-flan-t5-xl', save_to_disk_column=None, *args, **kwargs):
        super().setup(*args, **kwargs)

        self.pretrained_model_name = pretrained_model_name
        self.save_to_disk_column = save_to_disk_column
        
        self.data = EasyDict()
        self.config = self.global_config
        


    def _call(self, inputs):
        for input_data in inputs:
            self.data.update(input_data)
        
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import psutil
        self.processor = Blip2Processor.from_pretrained(self.pretrained_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.pretrained_model_name, torch_dtype=self.dtype
        )
        self.model.eval()
        self.model.to(self.device)

        
        def _caption_with_blip(batch_examples):

            flag_all_prepared = False

            # if all captions in this batch have been prepared
            if self.save_to_disk_column is not None:
                flag_all_prepared = True
                paths_to_save = batch_examples[self.save_to_disk_column]
                paths_to_save = [
                    p + '.txt' for p in paths_to_save
                ]
                for p in paths_to_save:
                    if not os.path.exists(p):
                        flag_all_prepared = False
            

            if flag_all_prepared:
                print("skipped generation!")
                generated_text = []
                # Read and return these captions
                for p in paths_to_save:
                    with open(p, 'r') as f:
                        caption = f.read()
                        generated_text.append(caption)
                batch_examples['img_caption'] = generated_text
            else:
                # read images
                input_images = [PIL.Image.open(img_path) for img_path in batch_examples['image_path']]
                inputs = self.processor(images=input_images, return_tensors='pt').to(self.device, dtype=self.dtype)
                generated_ids = self.model.generate(**inputs)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                batch_examples['img_caption'] = generated_text

            if self.save_to_disk_column is not None:
                for p, c in zip(paths_to_save, generated_text):
                    with open(p, 'w') as f:
                        f.write(c)
            
            # print(f"[Captioning] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
            return batch_examples

        output_data = DatasetDict()

        for data_split in self.splits_to_process:
            ds = self.data[data_split]
            ds_with_caption = ds.map(_caption_with_blip, batched=True, batch_size=128, writer_batch_size=128)
            output_data[data_split] = ds_with_caption
        
        self.model = self.model.cpu()
        del self.model
        print(output_data)
        return output_data
    

@register_transform_functor
class CaptionImageWithBLIP2v2(HFDatasetTransform):
    def setup(self, pretrained_model_name='Salesforce/blip2-flan-t5-xl', save_to_disk_column=None, index_name="image_captions", *args, **kwargs):
        super().setup(*args, **kwargs)

        self.pretrained_model_name = pretrained_model_name
        self.save_to_disk_column = save_to_disk_column
        
        self.data = EasyDict()
        self.config = self.global_config
        
        self.index_name = index_name


    def _call(self, inputs):
        for input_data in inputs:
            self.data.update(input_data)
        
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import psutil
        self.processor = Blip2Processor.from_pretrained(self.pretrained_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.pretrained_model_name, torch_dtype=self.dtype
        )
        self.model.eval()
        self.model.to(self.device)

        
        # Prepare ElasticSearch
        from elasticsearch import Elasticsearch, helpers

        # Password for the 'elastic' user generated by Elasticsearch
        ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]
        es = Elasticsearch(
            "https://localhost:9200",
            ca_certs=os.environ["ELASTIC_CA_CERTS"],
            basic_auth=("elastic", ELASTIC_PASSWORD),
        )

        index_name = self.index_name
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name)

        def _read_captions(batch_examples):
            es = Elasticsearch(
                "https://localhost:9200",
                ca_certs=os.environ["ELASTIC_CA_CERTS"],
                basic_auth=("elastic", ELASTIC_PASSWORD),
                timeout=60,
            )
            batch_indices = batch_examples['image_path']
            batch_indices = [f"{idx.split('/')[-1]}" for idx in batch_indices]
            # batch_cache_filenames = [
            #     os.path.join(self.cache_folder, f"{idx.split('/')[-1]}.safetensors") for idx in batch_indices
            # ]
            found = []
            captions = []

            queries = batch_indices
            docs = [
                {
                    '_index': index_name,
                    '_id': q,
                } for q in queries
            ]
            resp = es.mget(index=index_name, docs=docs)
            for doc in resp['docs']:
                if doc['found']:
                    captions.append(doc['_source']['caption'])
                    found.append(True)
                else:
                    captions.append("")
                    found.append(False)

            batch_examples['img_caption'] = captions
            batch_examples['found'] = found
            return batch_examples

        def _caption_with_blip(batch_examples):
            es = Elasticsearch(
                "https://localhost:9200",
                ca_certs=os.environ["ELASTIC_CA_CERTS"],
                basic_auth=("elastic", ELASTIC_PASSWORD),
                timeout=60,
            )
            batch_indices = batch_examples['image_path']
            batch_indices = [f"{idx.split('/')[-1]}" for idx in batch_indices]
            found = batch_examples['found']

            flag_all_prepared = (sum(found) == len(found))
            
            # if all captions in this batch have been prepared
            if self.save_to_disk_column is not None:
                paths_to_save = batch_examples[self.save_to_disk_column]
                paths_to_save = [
                    p + '.txt' for p in paths_to_save
                ]
            

            if flag_all_prepared:
                print("skipped generation!")
            else:
                # read images
                input_images = [PIL.Image.open(img_path) for img_path in batch_examples['image_path']]
                inputs = self.processor(images=input_images, return_tensors='pt').to(self.device, dtype=self.dtype)
                generated_ids = self.model.generate(**inputs)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                batch_examples['img_caption'] = generated_text

                # if self.save_to_disk_column is not None:
                #     # save to folder as backup
                #     for p, c in zip(paths_to_save, generated_text):
                #         with open(p, 'w') as f:
                #             f.write(c)
                
                actions = []

                for file_id, is_found, c in zip(batch_indices, found, generated_text):
                    
                    action = {
                        '_op_type': "index",
                        '_index': index_name,
                        '_id': file_id,
                        '_source': {
                            'caption': c,
                        }
                    }
                    if not is_found:
                        actions.append(action)
                
                if len(actions) > 0:
                    res = helpers.bulk(es, actions, request_timeout=120)
                    # print(f"number of success {res[0]}")
                    if res[0] != len(actions):
                        print("errors", res[1])
                    print(f"Successfully indexed {len(actions)} items into ES.")
                else:
                    print("No new items to index.")
                
            # print(f"[Captioning] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
            return batch_examples

        output_data = DatasetDict()

        for data_split in self.splits_to_process:
            ds = self.data[data_split]
            ds = ds.map(_read_captions, batched=True, batch_size=1000)
            # sort by found
            ds = ds.sort('found', reverse=True)
            ds_with_caption = ds.map(_caption_with_blip, batched=True, batch_size=128, writer_batch_size=128)
            output_data[data_split] = ds_with_caption
        
        self.model = self.model.cpu()
        del self.model
        print(output_data)
        return output_data

@register_transform_functor
class CaptionImageWithBLIP2v3(HFDatasetTransform):
    def setup(self, pretrained_model_name='Salesforce/blip2-flan-t5-xl', save_to_disk_column=None, index_name="image_captions", *args, **kwargs):
        super().setup(*args, **kwargs)

        self.pretrained_model_name = pretrained_model_name
        self.save_to_disk_column = save_to_disk_column
        
        self.data = EasyDict()
        self.config = self.global_config
        
        self.index_name = index_name


    def _call(self, inputs):
        for input_data in inputs:
            self.data.update(input_data)
        
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import psutil
        self.processor = Blip2Processor.from_pretrained(self.pretrained_model_name)
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.pretrained_model_name, torch_dtype=self.dtype
        )
        self.model.eval()
        
        from multiprocess import set_start_method
        try:
            set_start_method('spawn', force=True)
            print("spawned")
        except RuntimeError:
            pass
        
        # Prepare ElasticSearch
        from elasticsearch import Elasticsearch, helpers

        # Password for the 'elastic' user generated by Elasticsearch
        ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]
        es = Elasticsearch(
            "https://localhost:9200",
            ca_certs=os.environ["ELASTIC_CA_CERTS"],
            basic_auth=("elastic", ELASTIC_PASSWORD),
        )

        index_name = self.index_name
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name)

        def _read_captions(batch_examples):
            es = Elasticsearch(
                "https://localhost:9200",
                ca_certs=os.environ["ELASTIC_CA_CERTS"],
                basic_auth=("elastic", ELASTIC_PASSWORD),
                timeout=60,
            )
            batch_indices = batch_examples['image_path']
            batch_indices = [f"{idx.split('/')[-1]}" for idx in batch_indices]
            # batch_cache_filenames = [
            #     os.path.join(self.cache_folder, f"{idx.split('/')[-1]}.safetensors") for idx in batch_indices
            # ]
            found = []
            captions = []

            queries = batch_indices
            docs = [
                {
                    '_index': index_name,
                    '_id': q,
                } for q in queries
            ]
            resp = es.mget(index=index_name, docs=docs)
            for doc in resp['docs']:
                if doc['found']:
                    captions.append(doc['_source']['caption'])
                    found.append(True)
                else:
                    captions.append("")
                    found.append(False)

            batch_examples['img_caption'] = captions
            batch_examples['found'] = found
            return batch_examples

        def _caption_with_blip(batch_examples, rank, vision_model, image_processor):
            
            es = Elasticsearch(
                "https://localhost:9200",
                ca_certs=os.environ["ELASTIC_CA_CERTS"],
                basic_auth=("elastic", ELASTIC_PASSWORD),
                timeout=60,
            )
            batch_indices = batch_examples['image_path']
            batch_indices = [f"{idx.split('/')[-1]}" for idx in batch_indices]
            found = batch_examples['found']

            flag_all_prepared = (sum(found) == len(found))
            
            # if all captions in this batch have been prepared
            if self.save_to_disk_column is not None:
                paths_to_save = batch_examples[self.save_to_disk_column]
                paths_to_save = [
                    p + '.txt' for p in paths_to_save
                ]
            

            if flag_all_prepared:
                print("skipped generation!")
            else:
                device = torch.device(f"cuda:{rank}")
                vision_model = vision_model.to(device)

                # read images
                input_images = [PIL.Image.open(img_path) for img_path in batch_examples['image_path']]
                inputs = image_processor(images=input_images, return_tensors='pt').to(device, dtype=self.dtype)
                generated_ids = vision_model.generate(**inputs)
                generated_text = image_processor.batch_decode(generated_ids, skip_special_tokens=True)
                batch_examples['img_caption'] = generated_text

                # if self.save_to_disk_column is not None:
                #     # save to folder as backup
                #     for p, c in zip(paths_to_save, generated_text):
                #         with open(p, 'w') as f:
                #             f.write(c)
                
                actions = []

                for file_id, is_found, c in zip(batch_indices, found, generated_text):
                    
                    action = {
                        '_op_type': "index",
                        '_index': index_name,
                        '_id': file_id,
                        '_source': {
                            'caption': c,
                        }
                    }
                    if not is_found:
                        actions.append(action)
                
                if len(actions) > 0:
                    res = helpers.bulk(es, actions, request_timeout=120)
                    # print(f"number of success {res[0]}")
                    if res[0] != len(actions):
                        print("errors", res[1])
                    print(f"[Rank {rank}] Successfully indexed {len(actions)} items into ES.")
                else:
                    print("No new items to index.")

                vision_model = vision_model.cpu()

            # print(f"[Captioning] Memory Usage {psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024:.4f} GB")
            return batch_examples

        output_data = DatasetDict()

        for data_split in self.splits_to_process:
            ds = self.data[data_split]
            ds = ds.map(_read_captions, batched=True, batch_size=1000)
            # sort by found
            ds = ds.sort('found', reverse=True)
            ds_with_caption = ds.map(
                _caption_with_blip, 
                batched=True, 
                batch_size=128, 
                with_rank=True,
                num_proc=4,
                # remove_columns=["pixel_values"],
                fn_kwargs={
                    "vision_model": self.model,
                    "image_processor": self.processor,
                },
                writer_batch_size=128
            )
            output_data[data_split] = ds_with_caption
        
        del self.model
        print(output_data)
        return output_data
@register_transform_functor
class MergeDataColumns(HFDatasetTransform):
    def setup(self, pretrained_model_name='Salesforce/blip2-flan-t5-xl', save_to_disk_column=None, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs):

        module_config = self.module_config

        merge_on = module_config.merge_on
        merge_from = module_config.merge_from

        output_data = DatasetDict()

        for data_split in self.splits_to_process:
            ds = inputs[merge_on][data_split]
            ds_from = inputs[merge_from][data_split]
            print("merge on <-- ", ds)
            print("merge from <-- ", ds_from)
            df = ds.to_pandas()#.drop('__index_level_0__', axis=1)
            df_from = ds_from.to_pandas()#.drop('__index_level_0__', axis=1)
            # print("merge on <-- ", df)
            # print("merge from <-- ", df_from)
            column_to_join_on = 'question_id'
            cols_to_use = list(df_from.columns.difference(df.columns)) + [column_to_join_on]
            print('cols_to_use --> ', cols_to_use)
            df_merged = df.merge(df_from[cols_to_use], on=column_to_join_on, how="left")
            print(df_merged.head())
            ds_merged = Dataset.from_pandas(df_merged)
            print("merged --> ", ds_merged)
            output_data[data_split] = ds_merged
        print(output_data)
        return output_data
    
@register_transform_functor
class ShuffleData(HFDatasetTransform):
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.config = self.global_config

    def _call(self, inputs):
        full_dataset = []
        split_num = []
        for data_split in self.splits_to_process:
            ds = inputs[data_split]
            full_dataset.append(ds)
            split_num.append(len(ds))
        
        full_dataset = concatenate_datasets(full_dataset)
        full_dataset = full_dataset.shuffle(seed=42)
        print("shuffled dataset", full_dataset)
        split_indices = [0] + list(np.cumsum(split_num))
        print("split_indices", split_indices)
        output_data = DatasetDict()
        for i in range(len(self.splits_to_process)):
            data_split = self.splits_to_process[i]
            ds = full_dataset.select(range(split_indices[i], split_indices[i+1]))
            output_data[data_split] = ds
        
        return output_data

