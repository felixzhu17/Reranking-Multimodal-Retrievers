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


from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import CheckpointIO


# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from src.models.retriever.visual_colbert import VisualColBERTForPretraining
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from functools import partial
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher

from src.metrics import MetricsProcessor
from src.models.retriever.retriever_dpr import RetrieverDPR
from src.utils.dirs import *
import faiss
import wandb


import logging
logger = logging.getLogger(__name__)


from src.executors.ColBERT_vision_pretraining_executor import ColBERTVisionPretrainingExecutor

@register_executor
class ColBERTVisionLossInvestigationExecutor(ColBERTVisionPretrainingExecutor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        global_config=None,
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, global_config=global_config, use_data_node=use_data_node, *args, **kwargs)
        
        
    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        train_batch = {
            'Q': [
                sample_batched['input_ids'].to(self.device),
                sample_batched['attention_mask'].to(self.device)
            ],
            "D": [
                sample_batched['decoder_input_ids'].to(self.device),
                sample_batched['decoder_input_attention_mask'].to(self.device)
            ],
            # 'labels': sample_batched['labels'].to(self.device),
        }
        
        # if there is vision input, add it to the batch
        pixel_values = sample_batched.get('pixel_values', None)
        if pixel_values is not None:
            train_batch['Q'].append(pixel_values.to(self.device))
        
        image_features = sample_batched.get('image_features', None)
        if image_features is not None:
            train_batch['Q'].append(image_features.to(self.device))
        item_image_features = sample_batched.get('item_image_features', None)
        if item_image_features is not None:
            train_batch['D'].append(item_image_features.to(self.device))
        
        scores = self.model(**train_batch)
        
        config = self.model.colbert_config
        if config.use_ib_negatives:
            scores, ib_loss = scores
            loss = ib_loss
        else:
            scores = scores.view(-1, config.nway)
            labels = torch.zeros(sample_batched['input_ids'].shape[0]*dist.get_world_size(), dtype=torch.long, device=self.device)
            loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])
        
        batch_loss = loss
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("valid/loss", batch_loss, on_step=True, logger=True, sync_dist=True)
        self.log("valid/ib_loss", ib_loss, on_step=True, logger=True, sync_dist=True)
        
        data_to_return = {
            'loss': batch_loss.detach().cpu().numpy(),
        }
        
        self.validation_step_outputs[dataloader_idx].append(data_to_return)
        
        return data_to_return
        

    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name, dataloader_idx=0, mode='test'):
        # Batching every validation step outputs
        # n_queries x hidden_size
        batch_loss = []
        for step_output in step_outputs:
            batch_loss.append(step_output['loss'])
        
        log_dict = EasyDict()
        log_dict.metrics = {
            'loss': float(np.mean(np.array(batch_loss))),
        }
        
        return log_dict

    def logging_results(self, log_dict, prefix='test'):
        
        ### Add test results to wandb / tensorboard
        metrics_to_log = EasyDict()
        
        # Refractor the column names
        for metric, value in log_dict.metrics.items():
            metrics_to_log[f'{prefix}/{metric}'] = value
        
        pprint(metrics_to_log)
        
        logger.info(f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}")
        
        # if self.trainer.state.stage in ['sanity_check'] and not self.perform_zero_shot_eval:
        #     logging.warning('Sanity check mode, not saving to loggers.')
        #     return
        
        # Add to loggers
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(metric, float(value), logger=True, sync_dist=True)
            else:
                logger.info(f'{metric} is not a type that can be logged, skippped.')

        