import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import json
import operator

import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import os.path
from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from runway_for_ml.utils.util import batch_depad
from torch.utils.data import DataLoader
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)
import random

from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import CheckpointIO
from transformers import AdamW, Adafactor, get_scheduler
from torch.optim import Adam

from functools import partial

from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from src.models.flmr import (
    FLMRConfig,
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
)
from src.models.flmr import index_custom_collection
from src.models.flmr import search_custom_collection, create_searcher
from src.models.rerank.rerank_model import RerankModel, FullContextRerankModel
from src.models.rerank.decoder_rerank_model import DecoderRerankModel, DecoderHeadRerankModel
from src.models.rerank.interaction_rerank_model import InteractionRerankModel

from src.metrics import MetricsProcessor
from src.utils.dirs import *
import faiss
import wandb
import GPUtil
import pickle

import logging

logger = logging.getLogger(__name__)

import datasets


def get_world_size():
    return dist.get_world_size()


@register_executor
class RerankerBaseExecutor(BaseExecutor, MetricsProcessor):
    def __init__(
        self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode,  # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        global_config=None,
        only_encode_item_once=False,
        load_only_vision_projection_weights=False,
        perform_zero_shot_eval=False,
        validation_indexing_source=None,
        index_splits=["train", "valid", "test"],
        split_to_retrieve_in_validation=None,
        use_index=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            data_pipeline_config,
            model_config,
            mode,
            train_config=train_config,
            test_config=test_config,
            log_file_path=log_file_path,
            global_config=global_config,
            use_data_node=use_data_node,
            *args,
            **kwargs,
        )

        # When this flag is set to True, we only encode the item once in non- sanity check mode
        self.only_encode_item_once = only_encode_item_once

        # When this flag is set to true, only parameters in vision_projection will be loaded
        self.load_only_vision_projection_weights = load_only_vision_projection_weights

        # When this flag is set to true, we will perform zero-shot evaluation at sanity check
        self.perform_zero_shot_eval = perform_zero_shot_eval

        # When a list of names are provided, the indexing process will be run multiple times
        # this allows for evaluating the validation sets on different corpora
        self.validation_indexing_source = validation_indexing_source

        # For VQA datasets, it might be overwhelming to index all the data in the training set. Change this list to index only a subset of the data.
        self.index_splits = index_splits

        # Whether to use custom index and skip embedding generation
        self.use_index = use_index

        if self.config.mode == "train":
            self.split_to_retrieve_in_validation = (
                split_to_retrieve_in_validation or "valid"
            )
        else:
            self.split_to_retrieve_in_validation = (
                split_to_retrieve_in_validation or "test"
            )

        self.validation_step_outputs = defaultdict(list)
        self.validation_batch = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.test_batch = defaultdict(list)

        self.init_retrieve()

    def _check_reranker_class(self, RerankerClass):
        # Define a dictionary of checks
        checks = {
            "decoder_reranker": (
                [DecoderRerankModel, DecoderHeadRerankModel],
                "Only DecoderRerankModel supports decoder_reranker",
                ["preflmr_attention_fusion", "interaction_reranker", "freeze_reranker_vision_encoder", "text_only"]
            ),
            "interaction_reranker": (
                [InteractionRerankModel],
                "Only InteractionRerankModel supports interaction_reranker",
                ["freeze_reranker_vision_encoder"]
            ),
            "full_context_reranker": (
                [FullContextRerankModel],
                "Only FullContextRerankModel",
                ["preflmr_attention_fusion", "interaction_reranker"]
            )
        }
        
        # Check each module in the dictionary
        for module, (valid_classes, error_message, forbidden_modules) in checks.items():
            if module in self.model_config.modules:
                assert RerankerClass in valid_classes, error_message
                for forbidden_module in forbidden_modules:
                    assert forbidden_module not in self.model_config.modules
                break
        else:
            # Default case if no modules in the dictionary are found
            assert RerankerClass == RerankModel, "Only RerankModel"
            
        # if "full_context_reranker" in self.model_config.modules:
        #     assert self.training_config.batch_size == 1, "FullContextReranker requires 1 batch size"

    def _init_model(self, model_config):
        """Initialize self.model

        Args:
            model_config (dict): contains key-values for model configuration
        """
        reranker_config = model_config.reranker_config
        RerankerClass = globals()[reranker_config.RerankerClass]
        self._check_reranker_class(RerankerClass)
        self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)
        
        if "weighted_regression" in self.model_config.modules:
            reranker_config.pos_weight = self.model_config.num_negative_samples + 1
        else:
            reranker_config.pos_weight = None
            
        
        self.reranker = RerankerClass(reranker_config)
        
        if "freeze_reranker_vision_encoder" in self.model_config.modules:
            print("Freezing Reranker vision encoders")
            for name, param in self.reranker.context_vision_encoder.named_parameters():
                param.requires_grad = False

        if "preflmr_attention_fusion" in self.model_config.modules or "interaction_reranker" in self.model_config.modules:
            retriever_config = model_config.retriever_config

            ModelClass = globals()[retriever_config.ModelClass]
            ConfigClass = globals()[retriever_config.ConfigClass]
            ModelVersion = retriever_config.ModelVersion

            config = ConfigClass.from_pretrained(ModelVersion, trust_remote_code=True)

            config.load_cpu_extension = True

            if retriever_config.ModelClass == "FLMRModelForRetrieval":
                flmr_query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                    ModelVersion, subfolder="query_tokenizer"
                )
                flmr_context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
                    ModelVersion, subfolder="context_tokenizer"
                )

                self.retriever = ModelClass.from_pretrained(
                    ModelVersion,
                    config=config,
                    query_tokenizer=flmr_query_tokenizer,
                    context_tokenizer=flmr_context_tokenizer,
                    torch_dtype=self.use_dtype,
                    trust_remote_code=True,
                )

            else:
                self.retriever = ModelClass.from_pretrained(ModelVersion, config=config, trust_remote_code=True)

            print("Freezing Retriever")
            for name, param in self.retriever.named_parameters():
                param.requires_grad = False

    def init_retrieve(self):

        import json

        # load all predictions in
        self.questionId2topPassages = {}
        for prediction_pkl in self.config.model_config.index_files.static_results:
            logger.info(f"Loading static retrieval results from {prediction_pkl}")
            if prediction_pkl.endswith(".json"):
                # load using json
                with open(prediction_pkl, "r") as f:
                    predictions = json.load(f)["output"]
                    for pred in predictions:
                        q_id = pred["question_id"]
                        top_ranking_passages = pred["top_ranking_passages"]
                        self.questionId2topPassages[q_id] = top_ranking_passages
            else:
                # Can use `src/tools/reduce_retrieval_result_file_size.py` to reduce json file size to speed up the loading
                # in this case, we load from a pkl file
                with open(prediction_pkl, "rb") as f:
                    predictions = pickle.load(f)["output"]
                    for pred in predictions:
                        q_id = pred["question_id"]
                        top_ranking_passages = pred["top_ranking_passages"]
                        self.questionId2topPassages[q_id] = top_ranking_passages
        logger.info(
            f"Loaded {len(self.questionId2topPassages)} static retrieval results."
        )

    def prepare_data(self):
        super().prepare_data()

    def setup(self, stage):
        super().setup(stage)
        self.prepared_data = self.dp.get_data([self.use_data_node], explode=True)
        print(len(self.prepared_data.vqa_data_with_dpr_output.get("lookup", {})))
        if len(self.prepared_data.vqa_data_with_dpr_output.get("lookup", {})) == 0:
            self.prepared_data.vqa_data_with_dpr_output.lookup = {}
            print("Loading lookup table...")
            for data_split in self.index_splits:
                if data_split not in self.prepared_data.vqa_data_with_dpr_output:
                    continue
                ds_split = self.prepared_data.vqa_data_with_dpr_output[data_split]
                lookup_dict = (
                    ds_split.to_pandas()
                    .set_index("question_id", drop=False)
                    .to_dict(orient="index")
                )
                self.prepared_data.vqa_data_with_dpr_output.lookup.update(lookup_dict)

            if dist.is_initialized():
                print(f"Rank {dist.get_rank()} Done loading lookup table.")
            else:
                print("Lookup table loaded without distributed setup.")
        # if isinstance(self.prepared_data.train_passages, datasets.Dataset):
        # ds = self.prepared_data.train_passages
        test_ds = (
            self.prepared_data.valid_passages
            if self.split_to_retrieve_in_validation == "valid"
            else self.prepared_data.test_passages
        )
        self.prepared_data.passages = EasyDict(
            {
                "dataset": test_ds,
                "id2doc": {},
            }
        )

        if self.validation_indexing_source is not None:
            for name in self.validation_indexing_source:
                self.prepared_data[name] = EasyDict(
                    {
                        "id2doc": {},
                    }
                )

        logger.info(f"Preparing {len(test_ds)} passage data in id2doc...")
        test_df = test_ds.to_pandas()
        for _, entry_data in tqdm(
            test_df.iterrows(), total=len(test_ds), desc="formatting the test passages"
        ):
            k = entry_data["passage_id"]
            v = entry_data
            self.prepared_data.passages.id2doc[k] = v["passage_content"]
            if self.validation_indexing_source is not None:
                source_name = v["source_name"]
                if source_name in self.validation_indexing_source:
                    self.prepared_data[source_name].id2doc[k] = v["passage_content"]

        if self.validation_indexing_source is not None:
            for name in self.validation_indexing_source:
                logger.info(
                    f"passages from the source {name} has {len(self.prepared_data[name].id2doc)}"
                )

        logger.info(f"Passages prepared.")
        self.passage_id2doc = self.prepared_data["passages"].id2doc
        self.data_loaders = self.prepared_data["data_loaders"]

        self.train_dataloaders = list(self.data_loaders["train"].values())
        self.valid_dataloaders = list(self.data_loaders["valid"].values())
        self.test_dataloaders = list(self.data_loaders["test"].values())

        self.tokenizers = self.prepared_data["tokenizers"]

        self.tokenizer = self.tokenizers["tokenizer"]
        self.decoder_tokenizer = self.tokenizers["decoder_tokenizer"]

        checkpoint_to_load = self.global_config.train.get("load_model_path", "")

        # Resize the bert embedding space to accommodate special tokens
        logger.info(
            f"tokenizer lengths = {len(self.tokenizer)} and {len(self.decoder_tokenizer)}"
        )

        if not checkpoint_to_load or checkpoint_to_load == "":
            logger.warning("No checkpoint found. First time to train...")
        else:
            # We manually load the state dict
            logger.info(f"Loading from {checkpoint_to_load}")
            state_dict_from_ckpt = torch.load(
                checkpoint_to_load, map_location=self.device
            )["state_dict"]
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            if self.load_only_vision_projection_weights:
                pretrained_dict = {
                    k: v
                    for k, v in state_dict_from_ckpt.items()
                    if k in model_dict and "vision_projection" in k
                }
            else:
                pretrained_dict = {k: v for k, v in state_dict_from_ckpt.items()}
            # logger.info(f"Load the following parameters from the given checkpoint: {pretrained_dict.keys()}")
            # logger.info(f"Loading the following parameters into the current model: {pretrained_dict.keys()}")

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            self.load_state_dict(model_dict, strict=False)

    def configure_optimizers(self):
        """
        Return optimizers and schedulers, and optionally load state from checkpoint.
        """
        optimizer_name = self.optimizer_config["optimizer_name"]
        optimizer_params = self.optimizer_config.get("optimizer_params", {})

        optimization_parameters = [
            {
                "params": [
                    p
                    for n, p in self.reranker.named_parameters()
                    if "late_interaction_adapter" not in n and p.requires_grad
                ],
                "lr": optimizer_params.get(
                    "lr", 0.001
                ),  # Make sure to use get() to provide a default value
                "initial_lr": optimizer_params.get("lr", 0.001),
            },
            {
                "params": [
                    p
                    for n, p in self.reranker.named_parameters()
                    if "late_interaction_adapter" in n and p.requires_grad
                ],
                "lr": self.optimizer_config.get(
                    "mapping_network_lr", optimizer_params.get("lr", 0.001)
                ),
                "initial_lr": self.optimizer_config.get(
                    "mapping_network_lr", optimizer_params.get("lr", 0.001)
                ),
            },
        ]

        for group in optimization_parameters:
            logger.info(
                "#params: {}   lr: {}".format(len(group["params"]), group["lr"])
            )

        """define optimizer"""

        if optimizer_name == "AdamW":
            self.optimizer = AdamW(optimization_parameters, **optimizer_params)
        elif optimizer_name == "Adafactor":
            self.optimizer = Adafactor(optimization_parameters, **optimizer_params)
        elif optimizer_name == "Adam":
            self.optimizer = Adam(optimization_parameters, **optimizer_params)
        else:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        checkpoint_to_load = self.global_config.train.get("load_model_path", "")
        if checkpoint_to_load:
            checkpoint = torch.load(checkpoint_to_load, map_location=self.device)
            if "optimizer_states" in checkpoint:
                logger.info(f"Loading optimizer")
                self.optimizer.load_state_dict(checkpoint["optimizer_states"][0])

        num_warmup_steps = self.optimizer_config.get("scheduler_params", {}).get(
            "num_warmup_steps", 0
        )
        if self.optimizer_config.get("scheduler", None) == "linear":
            from transformers import get_linear_schedule_with_warmup

            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.optimizer_config.get("scheduler", None) == "cosine":
            t_total = self.training_config.trainer_paras.max_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, t_total, eta_min=1e-5, last_epoch=-1, verbose=False
            )
        else:
            from transformers import get_constant_schedule_with_warmup

            # Using constant scheduler
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                last_epoch=self.global_step,
            )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            },
        }

    def negative_sample_model_inputs(self, sample_batched, batch):
        local_random = random.Random()
        context_input_ids, context_attention_masks, context_docs = [], [], []
        for question_id, pos_item_ids in zip(
            sample_batched["question_ids"], sample_batched["pos_item_ids"]
        ):
            retrieved_docs = self.static_retrieve(question_id).retrieved_docs
            pos_item_ids = set(pos_item_ids)
            positive_docs = []
            negative_docs = []

            for doc in retrieved_docs:
                if doc["passage_id"] in pos_item_ids:
                    positive_docs.append(doc)
                else:
                    negative_docs.append(doc)

            if len(positive_docs) == 0:
                passage_id = local_random.sample(pos_item_ids, 1)[0]
                positive_docs = [
                    {
                        "score": 10,
                        "title": "",
                        "content": self.passage_id2doc.get(passage_id, ""),
                        "passage_id": passage_id,
                    }
                ]
            sampled_docs = local_random.sample(positive_docs, 1) + local_random.sample(
                negative_docs, self.model_config.num_negative_samples
            )
            retrieved_docs_content = [doc["content"] for doc in sampled_docs]
            context_input_id, context_attention_mask = self.tokenize_retrieved_docs(
                retrieved_docs_content
            )
            context_input_ids.append(context_input_id)
            context_attention_masks.append(context_attention_mask)
            context_docs.extend(retrieved_docs_content)
        if "decoder_reranker" in self.model_config.modules or "full_context_reranker" in self.model_config.modules:
            batch["context_text_sequences"] = context_docs
        else:
            batch["context_input_ids"] = torch.cat(context_input_ids, dim=0)
            batch["context_attention_mask"] = torch.cat(
                context_attention_masks, dim=0
            )
        return batch, None

    def sample_model_inputs(self, sample_batched, batch, n_docs = None):
        local_random = random.Random()
        context_input_ids, context_attention_masks, context_docs, labels = [], [], [], []
        n_docs = n_docs if n_docs else self.model_config.num_negative_samples + 1
        for question_id, pos_item_ids in zip(
            sample_batched["question_ids"], sample_batched["all_pos_item_ids"]
        ):
            retrieved_docs = self.static_retrieve(question_id).retrieved_docs
            pos_item_ids = set(pos_item_ids)

            # Sample documents
            sampled_docs = local_random.sample(retrieved_docs, n_docs)

            # Create labels for the sampled documents
            question_labels = [1 if doc["passage_id"] in pos_item_ids else 0 for doc in sampled_docs]

            # Extract content from sampled documents
            retrieved_docs_content = [doc["content"] for doc in sampled_docs]

            context_input_id, context_attention_mask = self.tokenize_retrieved_docs(retrieved_docs_content)
            context_input_ids.append(context_input_id)
            context_attention_masks.append(context_attention_mask)
            context_docs.extend(retrieved_docs_content)
            labels.extend(question_labels)
            
        if "decoder_reranker" in self.model_config.modules or "full_context_reranker" in self.model_config.modules:
            batch["context_text_sequences"] = context_docs
        else:
            batch["context_input_ids"] = torch.cat(context_input_ids, dim=0)
            batch["context_attention_mask"] = torch.cat(
                context_attention_masks, dim=0
            )
        batch["num_negative_examples"] = n_docs - 1
            
        return batch, labels

    def training_step(self, sample_batched, batch_idx):
        retrieval_results = None
        labels = None
        train_batch = self.get_model_inputs(sample_batched)

        if "train_with_retrieved_docs" in self.model_config.modules:
            if self.model_config.reranker_config.loss_fn == "negative_sampling":
                train_batch, labels = self.negative_sample_model_inputs(sample_batched, train_batch)
            else:
                train_batch, labels = self.sample_model_inputs(sample_batched, train_batch)

        if "interaction_reranker" in self.model_config.modules:
            retrieval_results = self.retriever(**train_batch)
            train_batch = {
                "query_late_interaction": retrieval_results.query_late_interaction_output,
                "context_late_interaction": retrieval_results.context_late_interaction_output,
                "num_negative_examples": self.model_config.num_negative_samples,
                "query_mask": retrieval_results.query_mask,
                "context_mask": retrieval_results.context_mask,
            }
            
                        
        if "preflmr_attention_fusion" in self.model_config.modules:
            train_batch["preflmr_scores"] = retrieval_results.scores_raw if retrieval_results is not None else self.retriever(**train_batch).scores_raw
            train_batch["fusion_multiplier"] = self.model_config.fusion_multiplier
            
        if "train_with_retrieved_docs" in self.model_config.modules and self.model_config.reranker_config.loss_fn != "negative_sampling":
            assert labels is not None
            train_batch["labels"] = labels
        else:
            assert labels is None
            
        if "text_only" in self.model_config.modules:
            train_batch["query_pixel_values"] = None
            
        batch_loss = self.reranker(**train_batch).loss

        # log the current learning rate from shedulers
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(
                f"train/lr[{index}]",
                current_lr,
                prog_bar=True,
                on_step=True,
                logger=True,
                sync_dist=True,
            )

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train/loss", batch_loss, on_epoch=True, on_step=True, logger=True, sync_dist=True)

        data_to_return = {
            "loss": batch_loss,
        }
        
        return data_to_return

    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        pred, batch = self._compute_loss(sample_batched, batch_idx, dataloader_idx)
        self.validation_step_outputs[dataloader_idx].append(pred)
        self.validation_batch[dataloader_idx].append(batch)
        return pred

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        validation_batches = self.validation_batch
        if len(validation_step_outputs) == 0:
            return None
        for i in range(len(self.val_dataloader())):
            validation_step_output = validation_step_outputs[i]
            validation_batch = validation_batches[i]
            if "full_validation" in self.model_config.modules:
                log_dict = self.evaluate_outputs(validation_step_output, validation_batch)
            else:
                log_dict = self.fast_evaluate_outputs(validation_step_output, validation_batch)
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])
        self.validation_step_outputs = defaultdict(list)
        self.validation_batch = defaultdict(list)

        return None

    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        pred, batch = self._dummy_loss(sample_batched, batch_idx, dataloader_idx)
        # MODIFIED
        # pred, batch = self._compute_loss(sample_batched, batch_idx, dataloader_idx, n_docs=5)
        self.test_step_outputs[dataloader_idx].append(pred)
        self.test_batch[dataloader_idx].append(batch)
        return pred

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        #MODIFIED
        # with open('test_losses.pkl', 'wb') as f:
        #     pickle.dump(test_step_outputs, f)
        # raise ValueError
        
        test_batches = self.test_batch

        logger.info("reading global step of the checkpoint...")
        if self.trainer.ckpt_path is not None:
            self.ckpt_global_step = torch.load(
                self.trainer.ckpt_path, map_location=torch.device("cpu")
            )["global_step"]
        elif self.global_config.train.get("load_model_path", "") != "":
            self.ckpt_global_step = torch.load(
                self.global_config.train.load_model_path,
                map_location=torch.device("cpu"),
            )["global_step"]
        else:
            self.ckpt_global_step = self.global_step

        # self.save_HF_model()
        for i in range(len(self.test_dataloader())):
            test_step_output = test_step_outputs[i]
            test_batch = test_batches[i]
            # MODIFIED
            log_dict = self.evaluate_outputs(test_step_output, test_batch)
            # log_dict = self.fast_evaluate_outputs(test_step_output, test_batch)
            self.logging_results(
                log_dict,
                prefix=f"{self.config.test_suffix}_{self.test_dataloader_names[i]}",
            )

        self.test_step_outputs = defaultdict(list)
        self.test_batch = defaultdict(list)

        return None

    def _dummy_loss(self, sample_batched, batch_idx, dataloader_idx=0):
        batch_info = {
            "query_input_ids": sample_batched["input_ids"].to(self.device),
            "query_attention_mask": sample_batched["attention_mask"].to(self.device),
            "query_pixel_values": sample_batched["pixel_values"].to(self.device),
            "query_text_sequences": sample_batched["input_text_sequences"],
            "btach_idx": batch_idx,
            "question_ids": sample_batched["question_ids"],
            "questions": sample_batched["questions"],
            "pos_item_ids": sample_batched["all_pos_item_ids"],
            "neg_item_ids": sample_batched["neg_item_ids"],
        }
        
        data_to_return = {"loss": torch.tensor(0.0).to("cpu")}
        return data_to_return, batch_info

    def _compute_loss(self, sample_batched, batch_idx, dataloader_idx=0, n_docs = None):
        retrieval_results = None
        labels = None
        batch_info = {
            "query_input_ids": sample_batched["input_ids"].to(self.device),
            "query_attention_mask": sample_batched["attention_mask"].to(self.device),
            "query_pixel_values": sample_batched["pixel_values"].to(self.device),
            "query_text_sequences": sample_batched["input_text_sequences"],
            "btach_idx": batch_idx,
            "question_ids": sample_batched["question_ids"],
            "questions": sample_batched["questions"],
            "pos_item_ids": sample_batched["all_pos_item_ids"],
            "neg_item_ids": sample_batched["neg_item_ids"],
        }
        
        test_batch = self.get_model_inputs(sample_batched)
        
        if n_docs or "test_with_retrieved_docs" in self.model_config.modules:
            if self.model_config.reranker_config.loss_fn == "negative_sampling":
                print("HEREEEE")
                test_batch, labels = self.negative_sample_model_inputs(sample_batched, test_batch)
            else:
                test_batch, labels = self.sample_model_inputs(sample_batched, test_batch)
                    
        if "interaction_reranker" in self.model_config.modules:
            retrieval_results = self.retriever(**test_batch)
            test_batch = {
                "query_late_interaction": retrieval_results.query_late_interaction_output,
                "context_late_interaction": retrieval_results.context_late_interaction_output,
                "num_negative_examples": self.model_config.num_negative_samples,
                "query_mask": retrieval_results.query_mask,
                "context_mask": retrieval_results.context_mask,
            }            
        if "preflmr_attention_fusion" in self.model_config.modules:
            test_batch["preflmr_scores"] = retrieval_results.scores_raw if retrieval_results is not None else self.retriever(**test_batch).scores_raw
            test_batch["fusion_multiplier"] = self.model_config.fusion_multiplier
            
        if n_docs or "test_with_retrieved_docs" in self.model_config.modules and self.model_config.reranker_config.loss_fn != "negative_sampling":
            assert labels is not None
            test_batch["labels"] = labels
        else:
            assert labels is None
            
        if "text_only" in self.model_config.modules:
            test_batch["query_pixel_values"] = None
            
        #MODIFIED
        batch_loss = self.reranker(**test_batch).loss

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("valid/loss", batch_loss, on_step=True, logger=True, sync_dist=True)

        data_to_return = {
            # "btach_idx": batch_idx,
            # "question_ids": sample_batched["question_ids"],
            # "questions": sample_batched["questions"],
            # "pos_item_ids": sample_batched["all_pos_item_ids"],
            # "neg_item_ids": sample_batched["neg_item_ids"],
            "loss": batch_loss.detach().cpu(),

        }

        return data_to_return, batch_info

    def fast_evaluate_outputs(self, step_outputs, current_batches, mode="test"):
        batch_loss = []
        for step_output in step_outputs:
            batch_loss.append(step_output["loss"])
        loss = float(np.mean(np.array(batch_loss)))
        test_table = wandb.Table(columns=[])
        return EasyDict(
            {
                "metrics": {"loss": loss},
                "artifacts": {"test_table": test_table},
            }
        )


    def evaluate_outputs(self, step_outputs, current_batches, mode="test"):
        (
            question_ids,
            pos_item_ids,
            neg_item_ids,
            questions,
            batch_loss,
            query_input_ids,
            query_attention_masks,
            query_pixel_values,
            query_text_sequences
        ) = self.process_step_outputs_and_batches(step_outputs, current_batches)


        Ks = self.model_config.Ks
        max_K = max(Ks)
        assert (
            self.config.model_config.docs_to_rerank == max_K
        ), "The number of retrieved documents must be equal to the maximum K."

        print("Reranking the top retrieved documents...")
        batch_result = []
        for (
            question_id,
            pos_ids,
            neg_ids,
            query_input_id,
            query_attention_mask,
            query_pixel_value,
            query_text_sequence
        ) in tqdm(zip(
            question_ids,
            pos_item_ids,
            neg_item_ids,
            query_input_ids,
            query_attention_masks,
            query_pixel_values,
            query_text_sequences
        )):
            
            retrieval_results = None
            
            retrieved_docs = self.static_retrieve(question_id).retrieved_docs
            retrieved_docs_content = [i["content"] for i in retrieved_docs]
            if self.model_config.reranker_config.loss_fn == "negative_sampling":
                labels = None
            else:
                labels = [1 if doc["passage_id"] in pos_ids else 0 for doc in retrieved_docs]
            context_input_ids, context_attention_masks = self.tokenize_retrieved_docs(
                retrieved_docs_content
            )
            
            if "decoder_reranker" in self.model_config.modules or "full_context_reranker" in self.model_config.modules:
                batch_input = {
                    "query_text_sequences": [query_text_sequence],
                    "query_pixel_values": query_pixel_value,
                    "context_text_sequences": retrieved_docs_content,
                    "num_negative_examples": context_input_ids.shape[0] - query_input_id.shape[0],
                    "labels": labels
                }
                    

            else:
                batch_input = {
                    "query_input_ids": query_input_id,
                    "query_attention_mask": query_attention_mask,
                    "query_pixel_values": query_pixel_value,
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_masks,
                    "num_negative_examples": context_input_ids.shape[0] - query_input_id.shape[0],
                    
                }

            
                if "interaction_reranker" in self.model_config.modules:
                    retrieval_results = self.retriever(**batch_input)
                    batch_input = {
                        "query_late_interaction": retrieval_results.query_late_interaction_output,
                        "context_late_interaction": retrieval_results.context_late_interaction_output,
                        "num_negative_examples": retrieval_results.context_late_interaction_output.size(0) - retrieval_results.query_late_interaction_output.size(0),
                        "query_mask": retrieval_results.query_mask,
                        "context_mask": retrieval_results.context_mask,
                    }
                    
                    
                if "preflmr_attention_fusion" in self.model_config.modules:
                    preflmr_scores = retrieval_results.scores_raw if retrieval_results is not None else self.retriever(**batch_input).scores_raw
                    batch_input["preflmr_scores"] = preflmr_scores
                    batch_input["fusion_multiplier"] = self.model_config.fusion_multiplier
                    
                if "text_only" in self.model_config.modules:
                    batch_input["query_pixel_values"] = None
                    
                batch_input["labels"] = labels


            outputs = self.reranker(
                **batch_input
            )
            loss, all_logits = outputs.loss.detach().cpu().item(), outputs.logits

            # Detach and clone the logits to avoid modifying the computation graph
            all_logits = all_logits.clone().detach()
            logits_list = all_logits.squeeze().tolist()
            
            assert len(retrieved_docs) == len(
                logits_list
            ), "Length of retrieved_docs and all_logits must match."

            # Pair each doc with its corresponding logit and sort descending by logit
            doc_logits_pairs = list(zip(retrieved_docs, logits_list))
            sorted_docs = sorted(doc_logits_pairs, key=lambda x: x[1], reverse=True)

            raw_top_ranking_passages = [
                {
                    "passage_id": doc["passage_id"],
                    "content": doc["content"],
                    "score": None,
                }
                for doc in retrieved_docs
            ]

            top_ranking_passages = [
                {
                    "passage_id": doc["passage_id"],
                    "content": doc["content"],
                    "score": float(score),
                }
                for doc, score in sorted_docs
            ]

            query_item = self.prepared_data.vqa_data_with_dpr_output.lookup[
                str(question_id)
            ]
            # pos_item_contents = [self.prepared_data.passages.id2doc[pos_id] for pos_id in pos_ids]
            batched_data = {
                "question_id": question_id,
                "top_ranking_passages": top_ranking_passages,
                "raw_top_ranking_passages": raw_top_ranking_passages,
                "pos_item_ids": pos_ids,
                "neg_item_ids": neg_ids,
                "loss": loss
            }
            if query_item.get("answers", None) is not None:
                batched_data["answers"] = list(query_item.answers)
            if query_item.get("gold_answer", None) is not None:
                batched_data["gold_answer"] = query_item.gold_answer
            if query_item.get("question", None) is not None:
                batched_data["question"] = query_item.question
            batch_result.append(batched_data)

        print("Reranking done.")
        if self.config.args.log_prediction_tables_with_images:
            artifact = self.wandb_logger.experiment.use_artifact(
                self.config.args.wandb_artifacts, type="dataset"
            )

        # Log results
        columns = ["question_id", "input_image", "image_key", "pos_item_ids"] + [
            "p_{}".format(i) for i in range(max(Ks))
        ]
        test_table = wandb.Table(columns=columns)

        to_write_data = {
            "output": [],
        }
        for re in tqdm(batch_result):
            to_write_data["output"].append(re)
            question_id = re["question_id"]
            knowledge_item = self.prepared_data.vqa_data_with_dpr_output.lookup[
                str(question_id)
            ]

            # pos_item_contents = [self.prepared_data.passages.id2doc[pos_id] for pos_id in pos_ids]
            table_entry = [
                knowledge_item.get("img_id", knowledge_item.get("img_key_full")),
                knowledge_item["img_path"],
                knowledge_item["img_path"],
                str(knowledge_item["pos_item_ids"]),
                # pos_item_contents,
            ]

            # if self.config.args.log_prediction_tables_with_images:
            #     # Replace image keys with real images
            #     input_image_file_name = knowledge_item['img_file_name']
            #     input_image = artifact.get(input_image_file_name)
            #     if input_image is None:
            #         input_image = artifact.get(input_image_file_name)

            #     table_entry[1] = input_image

            table_entry += [p["content"] for p in re["top_ranking_passages"]]
            test_table.add_data(*table_entry)

        ##############################
        ##    Compute Metrics       ##
        ##############################
        data_used_for_metrics = EasyDict(
            mode=mode,
            epoch=self.current_epoch,
            batch_retrieval_result=batch_result,
            Ks=Ks,
        )

        log_dict = self.compute_metrics(data_used_for_metrics)

        log_dict.artifacts.test_table = test_table
        log_dict.artifacts.to_write_data = to_write_data

        log_dict.metrics.update(
            {
                "loss": float(np.mean(np.array([i['loss'] for i in batch_result]))),
            }
        )

        return log_dict

    def static_retrieve(self, question_id):

        n_docs = self.config.model_config.docs_to_rerank
        annotation = self.questionId2topPassages.get(str(question_id), None)
        if annotation is None:
            raise ValueError(
                f"Question {question_id} not found in the static retrieval results."
            )
        assert n_docs <= len(annotation), "Number of retrieved docs must be less than the total number of docs."
        top_passages = annotation[:n_docs]

        for p in top_passages:
            p["title"] = ""
            passage_id = p["passage_id"]
            p["content"] = self.passage_id2doc.get(passage_id, "")

        scores = [p["score"] for p in top_passages]
        scores = torch.FloatTensor(scores).to(device=self.device)

        return EasyDict(
            retrieved_docs=top_passages,
            doc_scores=scores,
        )

    def tokenize_retrieved_docs(self, retrieved_docs):
        encoding = self.decoder_tokenizer(
            retrieved_docs,
            padding="max_length",
            max_length=self.model_config.max_decoder_source_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        return input_ids, attention_mask

    def logging_results(self, log_dict, prefix="test"):

        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        artifacts_to_log = log_dict.artifacts
        wandb_artifacts_to_log = dict()
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f"{prefix}/{metric}"] = value

        # include other artifacts / metadata
        metrics_to_log[f"{prefix}/epoch"] = self.current_epoch
        wandb_artifacts_to_log.update(
            {
                f"predictions/step_{self.global_step}_MODE({self.config.mode})_SET({prefix})_rank({self.global_rank})": log_dict.artifacts[
                    "test_table"
                ]
            }
        )
        pprint(metrics_to_log)
        pprint(wandb_artifacts_to_log)

        logger.info(
            f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}"
        )

        if (
            self.trainer.state.stage in ["sanity_check"]
            and not self.perform_zero_shot_eval
        ):
            logging.warning("Sanity check mode, not saving to loggers.")
            return

        # Add to loggers
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(metric, float(value), logger=True, sync_dist=True)
            else:
                logger.info(f"{metric} is not a type that can be logged, skippped.")

        # Call wandb to log artifacts; remember to use commit=False so that the data will be logged
        #       with other metrics later.
        if self.config.args.log_prediction_tables:
            self.wandb_logger.experiment.log(wandb_artifacts_to_log, commit=False)

        if self.config.mode == "test":
            try: 
                from utils.numpy_encoder import NpEncoder

                # Save predictions to files for DPR-based VQA systems
                json_path = os.path.join(
                    self.config.test_dir,
                    f'{prefix.replace("/", "_")}_predictions_rank_{self.global_rank}.json',
                )
                with open(json_path, "w") as json_f:
                    json.dump(
                        artifacts_to_log.to_write_data, json_f, indent=4, cls=NpEncoder
                    )
                    logger.info("Predictions have been saved to {}".format(json_path))
            except AttributeError:
                pass

    def get_model_inputs(self, sample_batched):
        if "decoder_reranker" in self.model_config.modules or "full_context_reranker" in self.model_config.modules:
            return {
                "query_text_sequences": sample_batched["input_text_sequences"],
                "query_pixel_values": sample_batched["pixel_values"].to(self.device),
                "context_text_sequences": sample_batched["decoder_input_text_sequences"],
                "num_negative_examples": self.model_config.num_negative_samples,
            }
        else:
            return {
                "query_input_ids": sample_batched["input_ids"].to(self.device),
                "query_attention_mask": sample_batched["attention_mask"].to(self.device),
                "context_input_ids": sample_batched["decoder_input_ids"].to(self.device),
                "context_attention_mask": sample_batched["decoder_input_attention_mask"].to(
                    self.device
                ),
                "query_pixel_values": sample_batched["pixel_values"].to(self.device),
                "num_negative_examples": self.model_config.num_negative_samples,
            }
            
    def process_step_outputs_and_batches(self, step_outputs, current_batches):
        question_ids = []
        pos_item_ids = []
        neg_item_ids = []
        questions = []
        batch_loss = []
        query_input_ids = []
        query_attention_masks = []
        query_pixel_values = []
        query_text_sequences = []

        for step_output in step_outputs:
            batch_loss.append(step_output["loss"])


        for current_batch in current_batches:
            question_ids += current_batch["question_ids"]
            pos_item_ids.extend(current_batch["pos_item_ids"])
            neg_item_ids.extend(current_batch["neg_item_ids"])
            questions.extend(current_batch["questions"])
            split_input_ids = current_batch["query_input_ids"].split(1, dim=0)
            split_attention_masks = current_batch["query_attention_mask"].split(1, dim=0)
            split_pixel_values = current_batch["query_pixel_values"].split(1, dim=0)

            # Extend the lists with the newly split tensors
            query_input_ids.extend(split_input_ids)
            query_attention_masks.extend(split_attention_masks)
            query_pixel_values.extend(split_pixel_values)
            query_text_sequences.extend(current_batch["query_text_sequences"])

        assert (
            len(question_ids)
            == len(pos_item_ids)
            == len(neg_item_ids)
            == len(questions)
            == len(query_input_ids)
            == len(query_attention_masks)
            == len(query_pixel_values)
            == len(query_text_sequences)
        ), "All lists must have the same length."

        return (
            question_ids,
            pos_item_ids,
            neg_item_ids,
            questions,
            batch_loss,
            query_input_ids,
            query_attention_masks,
            query_pixel_values,
            query_text_sequences
        )
