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

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict
from datasets import concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging

logger = logging.getLogger(__name__)

from src.utils.dirs import create_dirs
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import PIL.Image

from datasets.utils.file_utils import get_datasets_user_agent
import requests
import hashlib


@register_transform_functor
class LoadPreprocessedData(HFDatasetTransform):
    """
    This functor loads CC data
    'process:LoadCCData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: false,
      setup_kwargs: {
        data_path: "/home/ubuntu/data/CC3M/LLaVA-CC3M-Pretrain-595K/PrepareCC595kDataForRetrieval.hf",
        passage_path: "/home/ubuntu/data/CC3M/LLaVA-CC3M-Pretrain-595K/SplitCC595kPassagesForLargeScaleTraining.hf",
      },
    },
    """

    def setup(
        self,
        data_path,
        passage_path,
        add_instruction=None,
        shuffle_splits=None,
        load_instruction=True,
        use_filtered_passages=False,
        num_data=None,
        num_passages=None,
        *args,
        **kwargs,
    ):
        self.data_path = data_path
        self.passage_path = passage_path
        self.num_passages = num_passages
        self.use_filtered_passages = use_filtered_passages
        self.load_instruction = load_instruction
        self.add_instruction = add_instruction
        self.num_data = num_data
        self.shuffle_splits = shuffle_splits
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, *args, **kwargs):
        res = DatasetDict()
        from datasets import disable_caching

        disable_caching()

        data = DatasetDict.load_from_disk(self.data_path)
        passages = DatasetDict.load_from_disk(self.passage_path)

        res["train"] = data["train"]
        res["valid"] = data["valid"] if "valid" in data.keys() else data["val"]

        if self.shuffle_splits is not None:
            for split in ["train", "valid"]:
                if split in self.shuffle_splits:
                    logger.info(f"Shuffling {split} split")
                    res[split] = res[split].shuffle(seed=42)

        if self.num_data is not None:
            for split in self.num_data.keys():
                logger.info(
                    f"Selecting {self.num_data[split]} examples from {split} split"
                )
                if self.num_data[split] != -1:
                    res[split] = res[split].select(range(self.num_data[split]))

        def add_instructions(example):
            # randomly sample instructions from self.add_instruction
            # add to "instruction" column
            sampled_instruction = random.choice(self.add_instruction)
            example["instruction"] = sampled_instruction
            return example

        if self.add_instruction is not None:
            for split in ["train", "valid"]:
                res[split] = res[split].map(
                    add_instructions, num_proc=32, load_from_cache_file=False
                )

        def combine_instruction_with_question(examples):
            combined = []
            if "question" not in examples.keys():
                questions = [""] * len(examples["question_id"])
            else:
                questions = examples["question"]
            for q, i in zip(questions, examples["instruction"]):
                i = i.strip()
                if i.endswith("."):
                    i = i[:-1]
                if q is None:
                    q = ""

                if i.endswith(":"):
                    new_question = f"{i} {q}".strip()
                else:
                    new_question = f"{i}: {q}".strip()
                combined.append(new_question)

            examples["question"] = combined
            return examples

        if self.load_instruction:
            for split in ["train", "valid"]:
                if "dummy" in res[split].column_names:
                    continue
                if "instruction" in res[split].column_names:
                    res[split] = res[split].map(
                        combine_instruction_with_question,
                        batched=True,
                        num_proc=32,
                        load_from_cache_file=False,
                    )
                else:
                    logger.warning(f"Instruction not found in {split} split, skipping.")

        res["passages"] = passages["passages"]
        res["filtered_passages"] = passages["filtered_passages"]

        if self.num_passages is not None:
            res["passages"] = res["passages"].select(range(self.num_passages))

        if self.use_filtered_passages:
            res["passages"] = res["filtered_passages"]

        for split in ["train", "valid"]:
            if "dummy" in res[split].column_names:
                continue
            if "img_path" not in res[split].column_names:
                logger.warning(
                    f"img_path not found in {split} split, adding img_path from image_path"
                )
                res[split] = res[split].add_column("img_path", res[split]["image_path"])

        print(res["train"][0])
        print(res["valid"][0])
        print(res["passages"][0])
        print(res["filtered_passages"][0])
        print(res)
        return res


@register_transform_functor
class LoadPreprocessedData_v2(HFDatasetTransform):
    """
    This functor loads CC data
    'process:LoadCCData': {
      transform_name: 'LoadPreprocessedData',
      regenerate: false,
      cache: false,
      setup_kwargs: {
        data_path: "/home/ubuntu/data/CC3M/LLaVA-CC3M-Pretrain-595K/PrepareCC595kDataForRetrieval.hf",
        passage_path: "/home/ubuntu/data/CC3M/LLaVA-CC3M-Pretrain-595K/SplitCC595kPassagesForLargeScaleTraining.hf",
      },
    },
    """

    def setup(
        self,
        data_path,
        passage_path,
        image_root_folder=None,
        add_instruction=None,
        shuffle_splits=None,
        load_instruction=True,
        num_data=None,
        num_passages=None,
        *args,
        **kwargs,
    ):
        self.data_path = data_path
        self.passage_path = passage_path
        self.image_root_folder = image_root_folder
        self.num_passages = num_passages
        self.load_instruction = load_instruction
        self.add_instruction = add_instruction
        self.num_data = num_data
        self.shuffle_splits = shuffle_splits
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, *args, **kwargs):
        res = DatasetDict()
        # from datasets import disable_caching
        # disable_caching()

        if "///" in self.data_path:
            # split and assign to each
            self.data_path, data_sub_folder = (
                self.data_path.split("///")[0],
                self.data_path.split("///")[1],
            )
            data = load_dataset(self.data_path, data_sub_folder)
        else:
            data = load_dataset(self.data_path)

        if "///" in self.passage_path:
            self.passage_path, passage_sub_folder = (
                self.passage_path.split("///")[0],
                self.passage_path.split("///")[1],
            )
            passages = load_dataset(self.passage_path, passage_sub_folder)
        else:
            passages = load_dataset(self.passage_path)

        res = data
        # if 'valid' not in res.keys() and 'test' not in res.keys():
        #     res['valid'] = res['train']
        #     res['test'] = res['train']

        all_splits = list(res.keys())

        if self.shuffle_splits is not None:
            for split in self.shuffle_splits:
                logger.info(f"Shuffling {split} split")
                res[split] = res[split].shuffle(seed=42)

        if self.num_data is not None:
            for split in self.num_data.keys():
                logger.info(
                    f"Selecting {self.num_data[split]} examples from {split} split"
                )
                if self.num_data[split] != -1:
                    res[split] = res[split].select(range(self.num_data[split]))

        def add_instructions(example):
            # randomly sample instructions from self.add_instruction
            # add to "instruction" column
            sampled_instruction = random.choice(self.add_instruction)
            example["instruction"] = sampled_instruction
            return example

        if self.add_instruction is not None:
            for split in all_splits:
                res[split] = res[split].map(
                    add_instructions, num_proc=32, load_from_cache_file=False
                )

        def combine_instruction_with_question(examples):
            combined = []
            if "question" not in examples.keys():
                questions = [""] * len(examples["question_id"])
            else:
                questions = examples["question"]
            for q, i in zip(questions, examples["instruction"]):
                i = i.strip()
                if i.endswith("."):
                    i = i[:-1]
                if q is None:
                    q = ""

                if i.endswith(":"):
                    new_question = f"{i} {q}".strip()
                else:
                    new_question = f"{i}: {q}".strip()
                combined.append(new_question)

            examples["question"] = combined
            return examples

        if self.load_instruction:
            for split in all_splits:
                if "dummy" in res[split].column_names:
                    continue
                if "instruction" in res[split].column_names:
                    res[split] = res[split].map(
                        combine_instruction_with_question,
                        batched=True,
                        num_proc=32,
                        keep_in_memory=False,
                        load_from_cache_file=False,
                    )
                else:
                    logger.warning(f"Instruction not found in {split} split, skipping.")

        for split in all_splits:
            if f"{split}_passages" in passages.keys():
                split_passages = passages[f"{split}_passages"]
                if self.num_passages is not None:
                    split_passages = split_passages.select(range(self.num_passages))
                res[f"{split}_passages"] = split_passages

        def concat_image_root_folder_with_img_path(example):
            example["img_path"] = os.path.join(
                self.image_root_folder, example["img_path"]
            )
            return example

        for split in all_splits:
            if "dummy" in res[split].column_names:
                continue
            if (
                "img_path" in res[split].column_names
                and self.image_root_folder is not None
            ):
                res[split] = res[split].map(
                    concat_image_root_folder_with_img_path,
                    num_proc=32,
                    load_from_cache_file=False,
                )
            # if 'img_path' not in res[split].column_names:
            #     logger.warning(f"img_path not found in {split} split, adding img_path from image_path")
            #     res[split] = res[split].add_column('img_path', res[split]['image_path'])

        print(res["train"][0])
        # print(res['test'][0])
        print(res["train_passages"][0])
        # print(res['test_passages'][0])
        print(res)
        return res


@register_transform_functor
class ConcatenatePassageDatasets(HFDatasetTransform):
    def setup(self, names, concat_splits=None, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config
        self.names = names
        self.concat_splits = concat_splits

    def _call(self, inputs, *args, **kwargs):
        """
        This function merges the passages of inputs
        """

        res = DatasetDict()

        for split in self.concat_splits:
            all_passages = []
            # merged_passages = None
            for index, merge_from_data in enumerate(inputs):
                print(merge_from_data.keys())
                passage_enabled = self.concat_splits[split][index]
                print(f"checking {split} {index}: {passage_enabled}")

                if passage_enabled:
                    if split == "valid_passages" and split not in merge_from_data:
                        # use test passages
                        test_ds = merge_from_data["test_passages"]
                        print(
                            f"using test passages for {split} since {split} not found in {merge_from_data}"
                        )
                        merge_from_data[split] = deepcopy(test_ds)
                    if "source_name" not in merge_from_data[split].column_names:
                        merge_from_data[split] = merge_from_data[split].add_column(
                            "source_name",
                            [self.names[index]] * len(merge_from_data[split]),
                        )

                    all_passages.append(merge_from_data[split])
                    # if merged_passages is None:
                    #     merged_passages = merge_from_data[split].to_pandas()
                    # else:
                    #     print(f"merging {len(merged_passages)}")
                    #     merged_passages = pd.concat([merged_passages, merge_from_data[split].to_pandas()], ignore_index=True)
                    # all_passages.append(merge_from_data[split].to_pandas())

            res[split] = concatenate_datasets(all_passages)

            # print("start merging passages")
            # for i in all_passages:
            #     print(i)
            #     print('------------------')

            # merged_passages = all_passages[0]
            # for i in all_passages[1:]:
            #     print(f"merging {len(i)}")
            #     merged_passages = pd.concat([merged_passages, i], ignore_index=True)

            # res[split] = Dataset.from_pandas(merged_passages)

        print("===================")
        print(res)
        print(res["train_passages"][0])
        print(res["test_passages"][0])
        return res


@register_transform_functor
class ConcatenateDatasets(HFDatasetTransform):
    def setup(self, concat_splits, negative_names, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.config = self.global_config
        self.concat_splits = concat_splits
        self.negative_names = negative_names

    def _call(self, inputs, *args, **kwargs):
        """
        This function reduces image dict to images that appear in the passages
        """
        merged_res = DatasetDict()
        for split in self.splits_to_process:
            print("processing split", split)
            all_split_data = []
            for input_data, enabled, negative_name in zip(
                inputs, self.concat_splits[split], self.negative_names
            ):
                print(f"checking {split}: {input_data.keys()} {enabled}")
                if split in input_data.keys() and enabled > 0:
                    if "dummy" in input_data[split].column_names:
                        raise ValueError(
                            f"dummy found in {input_data}, please disable this split in `concat_splits`."
                        )
                    input_data_split = input_data[split]
                    if enabled > 1:
                        # duplicate input_data_split
                        input_data_split = concatenate_datasets(
                            [input_data_split for _ in range(enabled)]
                        )
                    if enabled < 1 and enabled > 0:
                        print(
                            f"Randomly selecting {enabled} of {len(input_data_split)}"
                        )
                        # randomly select a subset
                        input_data_split = input_data_split.shuffle(seed=42).select(
                            range(int(len(input_data_split) * enabled))
                        )

                    print(input_data_split)
                    if "img_caption" in input_data_split.column_names:
                        input_data_split = input_data_split.remove_columns(
                            ["img_caption"]
                        )
                    if "img_id" not in input_data_split.column_names:
                        input_data_split = input_data_split.add_column(
                            "img_id", [None] * len(input_data_split)
                        )
                    if negative_name is not None and negative_name != "":
                        input_data_split = input_data_split.add_column(
                            "use_negative_items",
                            [negative_name] * len(input_data_split),
                        )
                        # print(input_data_split)
                        # input("here!")

                    all_split_data.append(input_data_split)

            # convert to pands
            all_split_data = [i.to_pandas() for i in all_split_data]
            print(all_split_data)
            # concat
            merged_split_data = pd.concat(all_split_data, ignore_index=True)
            print("merged -->")
            print(merged_split_data)
            # convert back to dataset
            merged_res[split] = Dataset.from_pandas(merged_split_data)

        print(merged_res)
        return merged_res


@register_transform_functor
class AddTextBasedVision(HFDatasetTransform):
    def setup(self, caption_config, object_config, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.config = self.global_config
        self.caption_config = caption_config
        self.object_config = object_config

    def _call(self, inputs, *args, **kwargs):
        """
        This function reduces image dict to images that appear in the passages
        """

        def add_text_based_vision(example):
            sample = EasyDict(example)

            module = self.caption_config
            if module is not None:
                if isinstance(sample.img_caption, dict):
                    caption = sample.img_caption["caption"]
                else:
                    caption = sample.img_caption
                text_sequence = " ".join(
                    [module.separation_tokens.start]
                    + [caption]
                    + [module.separation_tokens.end]
                )
                sample.question = " ".join([sample.question, text_sequence])

            module = self.object_config
            if module is not None:
                vision_sentences = []
                vision_sentences += [module.separation_tokens.start]
                for obj in sample.objects:
                    attribute_max = module.get("attribute_max", 0)
                    if attribute_max > 0:
                        # find suitable attributes
                        suitable_attributes = []
                        for attribute, att_score in zip(
                            obj["attributes"], obj["attribute_scores"]
                        ):
                            if (
                                att_score > module.attribute_thres
                                and len(suitable_attributes) < attribute_max
                            ):
                                suitable_attributes.append(attribute)
                        # append to the sentence
                        vision_sentences += suitable_attributes
                    vision_sentences.append(obj["class"])
                    vision_sentences.append(module.separation_tokens.sep)

                ocr = module.get("ocr", 0)
                if ocr > 0:
                    text_annotations = sample.img_ocr
                    filtered_descriptions = []
                    for text_annoation in text_annotations:
                        description = text_annoation["description"].strip()
                        description = description.replace(
                            "\n", " "
                        )  # remove line switching
                        # vision_sentences += [description]
                        # print('OCR feature:', description)
                        if description not in filtered_descriptions:
                            filtered_descriptions.append(description)
                    # print('OCR feature:', filtered_descriptions)
                    vision_sentences += filtered_descriptions

                vision_sentences += [module.separation_tokens.end]
                text_sequence = " ".join(vision_sentences)

                sample.question = " ".join([sample.question, text_sequence])

            return sample

        output_data = DatasetDict()
        for split in self.splits_to_process:
            if split in inputs.keys():
                print("processing split", split)
                ds_split = inputs[split]
                ds_split = ds_split.map(
                    add_text_based_vision, num_proc=32, load_from_cache_file=False
                )
                output_data[split] = ds_split

        print(output_data["train"][0])
        print(output_data)
        return output_data


@register_transform_functor
class AddInstruction(HFDatasetTransform):
    """
    This functor adds instruction to the question
    'process:AddInstructionToOKVQA': {
      transform_name: 'AddInstruction',
      input_node: 'process:PrepareWikipediaPassageAnnotationsForOKVQA',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        load_instruction: true,
        add_instruction: [
          "Using the provided image, obtain documents that address the subsequent question: ",
          "Retrieve documents that provide an answer to the question alongside the image: ",
          "Extract documents linked to the question provided in conjunction with the image: ",
          "Utilizing the given image, obtain documents that respond to the following question: ",
          "Using the given image, access documents that provide insights into the following question: ",
          "Obtain documents that correspond to the inquiry alongside the provided image: ",
          "With the provided image, gather documents that offer a solution to the question: ",
          "Utilizing the given image, obtain documents that respond to the following question: ",
        ],
      },
    },
    """

    def setup(self, load_instruction=True, add_instruction=None, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.load_instruction = load_instruction
        self.add_instruction = add_instruction
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs, *args, **kwargs):

        def add_instructions(example):
            # randomly sample instructions from self.add_instruction
            # add to "instruction" column
            sampled_instruction = random.choice(self.add_instruction)
            example["instruction"] = sampled_instruction
            return example

        res = inputs

        if self.add_instruction is not None:
            for split in self.splits_to_process:
                res[split] = res[split].map(
                    add_instructions, num_proc=32, load_from_cache_file=False
                )

        def combine_instruction_with_question(examples):
            combined = []
            if "question" not in examples.keys():
                questions = [""] * len(examples["question_id"])
            else:
                questions = examples["question"]
            for q, i in zip(questions, examples["instruction"]):
                i = i.strip()
                if i.endswith("."):
                    i = i[:-1]
                if q is None:
                    q = ""

                if i.endswith(":"):
                    new_question = f"{i} {q}".strip()
                else:
                    new_question = f"{i}: {q}".strip()
                combined.append(new_question)

            examples["question"] = combined
            return examples

        if self.load_instruction:
            for split in self.splits_to_process:
                if "instruction" in res[split].column_names:
                    res[split] = res[split].map(
                        combine_instruction_with_question,
                        batched=True,
                        num_proc=32,
                        load_from_cache_file=False,
                    )
                else:
                    logger.warning(f"Instruction not found in {split} split, skipping.")

        print(res["train"][0])
        print(res)
        return res
