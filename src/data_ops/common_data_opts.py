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
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer
from transformers import CLIPTokenizerFast
from transformers import ViTFeatureExtractor
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from transformers import AutoImageProcessor
from transformers import Blip2Processor

# from transformers import LlavaNextImageProcessor

# For ColBERT model
from colbert.modeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
    tensorize_triples,
)
from colbert.infra import Run, RunConfig, ColBERTConfig

from src.models.flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer

from src.models.custom_clip_processor import CustomCLIPImageProcessor

import logging

logger = logging.getLogger(__name__)

from src.data_ops.custom_datasets import *

torch.multiprocessing.set_sharing_strategy("file_system")


@register_transform_functor
class PrepareDataloaders(BaseTransform):
    def setup(
        self,
        datasets_config,
        tokenizer_config,
        extra_columns,
        pass_columns,
        *args,
        feature_extractor_config={},
        image_processor_config={},
        samplers_kwargs={},
        **kwargs,
    ):
        self.datasets_config = datasets_config
        self.tokenizer_config = tokenizer_config
        self.feature_extractor_config = feature_extractor_config
        self.image_processor_config = image_processor_config
        self.extra_columns = extra_columns
        self.pass_columns = pass_columns

        def _get_tokenizer(tokenizer_config):
            TokenizerClass = globals()[tokenizer_config.TokenizerClass]
            if TokenizerClass == QueryTokenizer or TokenizerClass == DocTokenizer:
                logger.info("initializing ColBERT tokenizers...")
                config = ColBERTConfig(
                    bsize=None,
                )
                config.checkpoint = "bert-base-uncased"
                tokenizer = TokenizerClass(config)

                SPECIAL_TOKENS = tokenizer_config.SPECIAL_TOKENS
                SPECIAL_TOKENS["additional_special_tokens"] = (
                    tokenizer.tok.additional_special_tokens
                    + SPECIAL_TOKENS["additional_special_tokens"]
                )
                tokenizer.tok.add_special_tokens(SPECIAL_TOKENS)

            else:
                if TokenizerClass == FLMRContextEncoderTokenizer:
                    tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
                        tokenizer_config.TokenizerModelVersion,
                        subfolder="context_tokenizer",
                        trust_remote_code=True,
                    )
                elif TokenizerClass == FLMRQueryEncoderTokenizer:
                    tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                        tokenizer_config.TokenizerModelVersion,
                        subfolder="query_tokenizer",
                        trust_remote_code=True,
                    )
                else:
                    tokenizer = TokenizerClass.from_pretrained(
                        tokenizer_config.TokenizerModelVersion, trust_remote_code=True
                    )

                SPECIAL_TOKENS = tokenizer_config.SPECIAL_TOKENS
                if TokenizerClass == Blip2Processor:
                    SPECIAL_TOKENS["additional_special_tokens"] = (
                        tokenizer.tokenizer.additional_special_tokens
                        + SPECIAL_TOKENS["additional_special_tokens"]
                    )
                    tokenizer.tokenizer.add_special_tokens(SPECIAL_TOKENS)
                else:
                    SPECIAL_TOKENS["additional_special_tokens"] = (
                        tokenizer.additional_special_tokens
                        + SPECIAL_TOKENS["additional_special_tokens"]
                    )
                    tokenizer.add_special_tokens(SPECIAL_TOKENS)

            return tokenizer

        self.tokenizers = {
            tokenizer_name: _get_tokenizer(tk_config)
            for tokenizer_name, tk_config in self.tokenizer_config.items()
        }

        def _get_feature_extractor(feature_extractor_config):
            if "FeatureExtractorClass" in feature_extractor_config:
                FeatureExtractorClass = globals()[
                    feature_extractor_config.FeatureExtractorClass
                ]
                return FeatureExtractorClass.from_pretrained(
                    feature_extractor_config.FeatureExtractorModelVersion
                )
            else:
                return None

        self.feature_extractors = {
            fe_name: _get_feature_extractor(fe_config)
            for fe_name, fe_config in self.feature_extractor_config.items()
        }

        def _get_image_processor(image_processor_config):
            if "ImageProcessorClass" in image_processor_config:
                ImageProcessorClass = globals()[
                    image_processor_config.ImageProcessorClass
                ]
                return ImageProcessorClass.from_pretrained(
                    image_processor_config.ImageProcessorModelVersion
                )
            else:
                return None

        self.image_processors = {
            name: _get_image_processor(cfg)
            for name, cfg in self.image_processor_config.items()
        }

        self.samplers_kwargs = samplers_kwargs

    def _call(self, inputs, *args, **kwargs):
        """_summary_

        Args:
            input_data (EasyDict): Corresponds to `self.data` object in `set_dataloader` function

        Returns:
            Dict: A dictionary of prepared DataLoader instances
        """
        logger.info("Running PrepareDataloaders")
        input_data = EasyDict()
        if isinstance(inputs, list):
            for i in inputs:
                input_data.update(i)
        else:
            input_data = inputs
        print(f"Received data columns: {input_data.keys()}")
        for k, v in input_data.items():
            print(f"{k}: {v}")

        output_data = EasyDict()
        prepared_dataloaders = EasyDict()
        for mode in self.datasets_config.keys():
            for data_config in self.datasets_config[mode]:
                if self.global_config.mode in ["train", "prepare_data"]:
                    if mode not in ["train", "valid"]:
                        prepared_dataloaders.setdefault(mode, {})
                        continue
                else:
                    if mode not in ["test"]:
                        prepared_dataloaders.setdefault(mode, {})
                        continue
                data_config = data_config.copy()
                use_column = data_config.pop("use_column")
                use_split = data_config.pop("split")
                dataset_type = data_config.pop("dataset_type")
                for i in input_data.keys():
                    try:
                        for j in input_data[i].keys():
                            dataset = input_data[i][j]
                            errors = []  # Initialize a list to collect error messages
                            count = (
                                0  # Initialize a counter to track the number of checks
                            )
                            for entry in dataset:
                                img_path = entry["img_path"]
                                if not os.path.exists(img_path):
                                    # Instead of raising an error immediately, collect the error message
                                    errors.append(
                                        f"Invalid image path in dataset '{i}[{j}]' at entry {count}: {img_path}"
                                    )
                                count += 1
                                if count >= 5:  # Break after checking 5 entries
                                    break
                            if (
                                errors
                            ):  # If there are any errors collected, raise an exception
                                error_message = "\n".join(errors)
                                raise ValueError(error_message)
                            print(
                                f"First 5 image paths in dataset '{i}[{j}]' of length {len(dataset)} are valid."
                            )
                    except Exception as e:
                        print(e)  # Print all collected errors
                        continue  # Continue to the next dataset

                dataset_dict = {
                    "data": input_data[use_column][use_split],
                    "tokenizers": self.tokenizers,
                    "feature_extractors": self.feature_extractors,
                    "image_processors": self.image_processors,
                    "mode": mode,
                    **data_config,
                }
                # Add extra data that is needed by datasets
                if isinstance(self.extra_columns, list):
                    for extra_column in self.extra_columns:
                        dataset_dict[extra_column] = input_data[extra_column]
                else:
                    for extra_column, extra_column_from in self.extra_columns.items():
                        print(
                            "extra_column",
                            extra_column,
                            "extra_column_from",
                            extra_column_from,
                        )
                        dataset_dict[extra_column] = input_data[extra_column_from]
                dataset = globals()[dataset_type](self.global_config, dataset_dict)

                if mode == "train":
                    sampler = RandomSampler(dataset)
                else:
                    sampler = RandomSampler(dataset)

                data_loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=self.global_config[mode].batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=self.global_config[mode].get(
                        "num_dataloader_workers", 0
                    ),
                )

                prepared_dataloaders.setdefault(mode, {})[
                    f"{mode}/{dataset_type}.{use_split}"
                ] = data_loader

                logger.info(
                    f"[Data Statistics]: {mode} data loader: {mode}/{dataset_type}.{use_split} {len(data_loader)}"
                )

        output_data = EasyDict(
            {
                "data_loaders": prepared_dataloaders,
                "tokenizers": self.tokenizers,
                "feature_extractors": self.feature_extractors,
            }
        )

        # Pass extra columns to trainers
        print(self.pass_columns)
        for extra_column, extra_column_from in self.pass_columns.items():
            output_data[extra_column] = input_data[extra_column_from]

        return output_data


@register_transform_functor
class WrapOutputIntoKeys(BaseTransform):
    def setup(self, *args, **kwargs):
        self.module_config = EasyDict(kwargs)
        self.data = EasyDict()
        self.config = self.global_config

    def _call(self, inputs):

        module_config = self.module_config

        output_keys = module_config.output_keys

        output_data = {}

        for output_key, input_ in zip(output_keys, inputs):
            output_data[output_key] = input_

        print("wrapped columns:", output_data.keys())
        return output_data
