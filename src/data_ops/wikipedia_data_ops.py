"""
This file defines the data transforms that will be applied to the data. 
Each transform takes in an EasyDict object of in_features (key: feature_name, value: feature data)
It should output an EasyDict object of out_features (key: feature_name, value: feature_data)
Each transform defined here can be used as an independent unit to form a data pipeline
Some common transforms are provided by runway
"""

from runway_for_ml.data_module.data_transforms import (
    BaseTransform,
    HFDatasetTransform,
    register_transform_functor,
    keep_ds_columns,
)

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

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging

logger = logging.getLogger(__name__)

from src.utils.dirs import create_dirs
from src.utils.vqa_tools import VQA
from src.utils.vqaEval import VQAEval

from transformers import (
    AutoImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from datasets import Dataset, load_from_disk
import PIL

from src.models.custom_clip_processor import CustomCLIPImageProcessor


@register_transform_functor
class LoadWikipediaPassageData(BaseTransform):
    def setup(self, add_title=False, *args, **kwargs):
        self.add_title = add_title
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, *args, **kwargs):
        """
        This function loads data from wiki_dpr Passage Corpus
        {
            "passage_data_path": {
                "train": "..",
                "full": "..",
            },
            "use_full_split": True,
        }
        """

        module_config = self.module_config
        self.data.passages = {
            "id2doc": {},  # full corpus
        }

        ######################
        # Read knowledge passage data
        ######################
        passage_file = module_config.passage_data_path.full
        ds = load_from_disk(passage_file)

        def add_title_to_passage(example):
            example["text"] = " ".join(
                ["title:", example["title"], "content:", example["text"]]
            )
            return example

        if self.add_title:
            ds = ds.map(add_title_to_passage, num_proc=32)

        self.data.passages = {
            "id2doc": {},
            "dataset": ds,
        }

        return self.data


@register_transform_functor
class LoadFullWikipediaPassageData(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()

    def _call(self, inputs, *args, **kwargs):
        """
        This function loads data from olm/wikipedia Passage Corpus
        {
            dataset_name: "olm/wikipedia",
            dataset_date: "20230301",
            truncation_length: 250,
        }
        """

        module_config = self.module_config
        self.data.passages = {
            "id2doc": {},  # full corpus
        }

        ######################
        # Read knowledge passage data
        ######################
        dataset_name = module_config.dataset_name
        dataset_date = module_config.dataset_date
        truncation_length = module_config.truncation_length

        ds = load_dataset(dataset_name, language="en", date=dataset_date)["train"]

        # a map function to process and split the text in batch
        def process_text(batch):
            final_batch = {
                "url": [],
                "text": [],
                "title": [],
                "id": [],
            }

            for index in range(len(batch["id"])):

                original_text = batch["text"][index]
                # print(index, original_text)
                # print('=====================')
                # if item['text'] is more than truncation length (approximately in tokens)
                # when counting the length, split the sentence by both space and newline

                # split the text by newline
                text_list = original_text.split("\n")
                # remove empty string
                text_list = [text for text in text_list if text != ""]
                # split the text by space
                text_list = [text.split() for text in text_list]
                # flatten the list
                text_list = [item for sublist in text_list for item in sublist]
                # print("flattened text_list: ", text_list)
                # get length of the text
                text_length = len(text_list)
                # print("text_length: ", text_length)
                # if the text is longer than truncation length, split the text at only "\n"
                if text_length > truncation_length:
                    # split the text by newline
                    text_list = original_text.split("\n")
                    # remove empty string
                    text_list = [text for text in text_list if text != ""]

                    # add items in text)_list gradually until the tuncation_length is met
                    # if the truncation_length is not met, add the next item in text_list
                    # if the truncation_length is met, refresh the final_text and add the next item in text_list
                    final_text = ""
                    for text in text_list:
                        # print("final_text: ", final_text)
                        if (
                            len(final_text.split()) + len(text.split())
                            < truncation_length
                        ):
                            final_text += text + "\n"
                        else:
                            final_batch["text"].append(final_text)
                            final_batch["url"].append(batch["url"][index])
                            final_batch["title"].append(batch["title"][index])
                            final_batch["id"].append(batch["id"][index])
                            final_text = ""

                    # if the final_text is not empty, add it to the final_batch
                    if final_text != "":
                        final_batch["text"].append(final_text)
                        final_batch["url"].append(batch["url"][index])
                        final_batch["title"].append(batch["title"][index])
                        final_batch["id"].append(batch["id"][index])

                else:
                    final_batch["text"].append(batch["text"][index])
                    final_batch["url"].append(batch["url"][index])
                    final_batch["title"].append(batch["title"][index])
                    final_batch["id"].append(batch["id"][index])

            return final_batch

        logger.info(f"Loaded {len(ds)} passages from {dataset_name} dataset")
        ds = ds.map(process_text, batched=True, num_proc=8)

        # reindex the dataset id column and remove the original id column
        ds = ds.rename_column("id", "original_id")
        ds = ds.add_column("id", range(len(ds)))
        ds = ds.remove_columns(["original_id"])
        logger.info(
            f"After processing, there are {len(ds)} passages from {dataset_name} dataset"
        )

        self.data.passages = {
            "id2doc": {},
            "dataset": ds,
        }

        return self.data


@register_transform_functor
class IndexPassagesWithElasticSearch(BaseTransform):
    def setup(self, _run_index=False, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config
        self.run_index = _run_index

    def _call(self, inputs, *args, **kwargs):
        """
        This function indexes passages into ElasticSearch
        {
            index_name: "wikipedia",
        },
        """
        for input_data in inputs:
            self.data.update(input_data)

        module_config = self.module_config

        if not self.run_index:
            logger.warning(
                "_run_index = False, Skipping indexing passages into ElasticSearch."
            )
            return self.data

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

        # delete the current index
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)

        all_actions = []
        for index, i in tqdm(enumerate(ds), total=len(ds)):
            # doc = {
            #     'title': i['title'],
            #     'text': i['text'],
            # }
            # resp = es.index(index="wikipedia", id=i['id'], document=doc)
            action = {
                "_op_type": "index",
                "_index": index_name,
                "_id": i["id"],
                "_source": {
                    "title": i["title"],
                    "text": i["text"],
                },
            }
            all_actions.append(action)
            # if index > 1000:
            #     break

        batch_size = 10000
        n_actions = len(all_actions)
        for i in range(0, n_actions, batch_size):
            print(f"processing...{i}-{i+batch_size}/{n_actions}")
            actions = all_actions[i : min(i + batch_size, n_actions)]

            res = helpers.bulk(es, actions, request_timeout=120)
            pprint(res)
            print(f"number of success {res[0]}")
            if res[0] != batch_size:
                print("errors", res[1])
        logger.info(f"Successfully indexed {n_actions} items into ES.")

        return self.data


@register_transform_functor
class PrepareWikipediaPassageAnnotations(HFDatasetTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function prepares Wikipedia passage annotations (pseudo labels)
        {
            "annotations_path": {
                "train": "..",
                "valid": "..",
                "test": "..",
            },
        },
        """
        for input_data in inputs:
            self.data.update(input_data)

        module_config = self.module_config

        ######################
        #  Get weak supervision annotations
        ######################
        self.data.okvqa_data_with_dpr_output = EasyDict(
            {
                "train": {},
                "test": {},
            }
        )
        annotations = EasyDict({})

        # Prepare ElasticSearch
        from elasticsearch import Elasticsearch, helpers

        # Password for the 'elastic' user generated by Elasticsearch
        ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]

        es = Elasticsearch(
            "https://localhost:9200",
            ca_certs=os.environ["ELASTIC_CA_CERTS"],
            basic_auth=("elastic", ELASTIC_PASSWORD),
            timeout=60,
        )

        # Successful response!
        es.info()

        ds = self.data.passages.dataset
        index_name = module_config.index_name

        output_data = DatasetDict()

        def search_for_a_string(query):
            resp = es.search(
                index=index_name,
                query={
                    "multi_match": {
                        "query": query,
                        "fields": ["title", "text"],
                        "type": "phrase",
                    }
                },
                timeout="60s",
            )
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

        from thefuzz import fuzz
        from thefuzz import process

        available_documents = {}

        # reindex the dataset id column and convert to strings. TODO: move this to the previous stage.
        # ds = ds.rename_column('id', 'original_id')
        # list_ids = [str(i) for i in range(len(ds))]
        # ds = ds.add_column('id', list_ids)
        # ds = ds.remove_columns(['original_id'])
        # cast id column to string
        # from datasets import Value
        # ds = ds.cast_column('id', Value(dtype='string', id=None))
        # ds = ds.rename_column('id', 'passage_id')

        for data_split in ["train", "test"]:
            missing_entries = []
            missing_data = []

            for item in tqdm(self.data.okvqa_data[data_split].data_items):
                question_id = item.question_id

                # Search ES and return all passages containing answers
                passages_match_answer = []

                for answer in set(item.answers):
                    passages_match_answer.extend(
                        search_for_a_string(answer)["hits"]["hits"]
                    )

                # Rate passages according to query information (e.g. question, objects in the image)
                choices = {
                    i["_id"]: i["_source"]["text"] for i in passages_match_answer
                }

                element_string_in_query = f'{item.gold_answer} {item.gold_answer} {item.question} {item.img_caption["caption"]}'

                for obj in item.objects:
                    element_string_in_query += f" {obj['class'].strip().lower()}"

                res = process.extract(
                    element_string_in_query,
                    choices,
                    limit=10,
                    scorer=fuzz.token_set_ratio,
                )
                # print("rating", choices, 'according to', item.question)
                # input()
                # drop lowest score item to further filter down the annotations
                if len(res) > 0:
                    lowest_score = res[-1][1]
                    res = [i for i in res if i[1] > lowest_score]
                else:
                    res = []

                retrieved_passages = {
                    i["_id"]: i["_source"] for i in passages_match_answer
                }

                knowledge_collection = []
                for i in res:
                    passage_data = {
                        "id": i[2],
                    }
                    passage_data.update(retrieved_passages[i[2]])
                    knowledge_collection.append(passage_data)

                # print("knowledge_collection", knowledge_collection)
                annotation = knowledge_collection

                if annotation is None:
                    missing_entries.append(str(question_id))
                    # logger.warning("question {} (split {}) not found in knowledge.".format(str(question_id), data_split))
                    if self.config.mode == "train":
                        continue
                    else:
                        # in testing mode, all samples must be used
                        pos_item_ids = ["1"]
                        related_item_ids = ["1"]
                        pos_item_contents = [""]
                else:
                    pos_item_ids = [p["id"] for p in annotation]
                    pos_item_contents = [
                        retrieved_passages[str(passage_id)]["text"]
                        for passage_id in pos_item_ids
                    ]
                    related_item_ids = list(retrieved_passages.keys())
                    if len(pos_item_ids) == 0:
                        missing_data.append(str(question_id))
                        # logger.warning("question {} (split {}) has no related knowledge in annotations.".format(str(question_id), data_split))
                        # related_knowledge = [1]
                        if self.config.mode == "train":
                            continue
                        else:
                            # in testing mode, all samples must be used
                            pos_item_ids = ["1"]
                            related_item_ids = ["1"]
                            pos_item_contents = [""]

                knowledge_item = EasyDict(dict(item))
                knowledge_item["pos_item_ids"] = pos_item_ids
                knowledge_item["pos_item_contents"] = pos_item_contents
                knowledge_item["related_item_ids"] = related_item_ids
                knowledge_item["question_id"] = str(question_id)
                self.data.okvqa_data_with_dpr_output[data_split][
                    str(question_id)
                ] = knowledge_item

            if len(missing_entries) > 0:
                logger.warning(
                    f"{len(missing_entries)} questions (split {data_split}) not found in knowledge. \n {missing_entries}"
                )
            if len(missing_data) > 0:
                logger.warning(
                    f"{len(missing_data)} questions (split {data_split}) has no annotations. \n {missing_data}"
                )

            # Load item data into lookup with question_id as index
            logger.info("Indexing data items...")

            # for item in tqdm(self.data.okvqa_data_with_dpr_output[data_split].data_items):
            #     question_id = item['question_id']
            #     self.data.okvqa_data_with_dpr_output.lookup[str(question_id)] = item

            ds_split = pd.DataFrame.from_dict(
                self.data.okvqa_data_with_dpr_output[data_split], orient="index"
            )

            ds_split = Dataset.from_pandas(ds_split)
            output_data[data_split] = ds_split

            # Report statistics
            logger.info(
                "[Data statistics] loaded with knowledge data split: {}  entries: {}".format(
                    data_split, len(ds_split)
                )
            )

        # OKVQA has only test sets
        output_data["valid"] = output_data["test"]

        print(output_data)
        return output_data


@register_transform_functor
class ReduceWikipediaPassagesSize(HFDatasetTransform):
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
        for data_split in ["train", "valid", "test"]:
            ds_split = self.data[data_split]
            all_related_item_ids = ds_split["related_item_ids"]
            for related_item_ids in all_related_item_ids:
                for doc_id in related_item_ids:
                    available_documents[doc_id] = 1

        print(f"total #docs {len(available_documents)}")

        # Randomly select self.num_random_passages passages
        if self.num_random_passages > 0:
            random_passages = random.sample(
                list(full_ds["id"]), self.num_random_passages
            )
            for doc_id in random_passages:
                available_documents[doc_id] = 1

        # filter out docs that don't have answers
        filtered_ds = full_ds.filter(
            lambda x: available_documents.get(x["id"], None) is not None,
            load_from_cache_file=False,
        )

        # rename id to passage_id, and text to passage_content
        filtered_ds = filtered_ds.rename_columns(
            {"id": "passage_id", "text": "passage_content"}
        )
        full_ds = full_ds.rename_columns(
            {"id": "passage_id", "text": "passage_content"}
        )

        output_data = DatasetDict(
            {
                "filtered_passages": filtered_ds,
                "passages": filtered_ds,
            }
        )
        logger.info(f"reducing #docs {len(full_ds)} (full) to {len(filtered_ds)}")
        print(output_data)
        return output_data
