import copy
import math
import os
from turtle import forward
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from easydict import EasyDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    T5PreTrainedModel,
)
from transformers import VisualBertModel, VisualBertConfig, BertTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from transformers import Blip2ForConditionalGeneration, Blip2Config
from src.models.retriever.retriever_dpr import RetrieverDPR

# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from src.models.retriever.visual_colbert import *
from colbert.modeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
    tensorize_triples,
)
from colbert.data import Queries
from colbert import Searcher

from transformers.models.rag.retrieval_rag import Index
from src.models.rerank.attention_fusion import AttentionFusionBertModel

import pytorch_lightning as pl

import time

import logging

logger = logging.getLogger(__name__)

import string
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
import faiss
import pickle
from typing import Iterable, List, Optional, Tuple
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import random
from src.models.custom_peft import PeftModelForSeq2SeqLM
from src.models.flmr.models.flmr.modeling_flmr import (
    FLMRConfig,
    FLMRQueryEncoderOutput,
    FLMRTextModel,
    FLMRVisionModel,
    FLMRMultiLayerPerceptron,
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
)
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import BertConfig

def initialise_loss_fn(config, device):
    if config.loss_fn == "BCE":
        pos_weight = torch.tensor([config.pos_weight], device=device) if config.pos_weight is not None else None
        if pos_weight is not None:
            print("Weighted BCE Loss", config.pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif config.loss_fn == "2H_BCE":
        class_weights = torch.tensor([1.0, config.pos_weight], device=device) if config.pos_weight is not None else None
        if class_weights is not None:
            print("Weighted CE Loss", class_weights)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    elif config.loss_fn == "negative_sampling":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function {config.loss_fn}")
    return loss_fn



def prepare_logits_labels(config, logits, logits_secondary, batch_size, num_negative_examples, labels=None):
    # Check if labels are provided
    if labels is not None:
        # Assert labels are a list
        assert isinstance(labels, list), "Labels must be a list"
        assert config.loss_fn != "negative_sampling", "Labels should not be provided for negative sampling loss function"
        # Convert labels to a PyTorch tensor and reshape to (-1, 1)
        labels = torch.tensor(labels, dtype=torch.float32, device = logits.device).reshape(-1, 1)

    if config.loss_fn in ["BCE", "2H_BCE"]:
        # First document is the positive example, concatenate them all along the first dimension and use binary cross entropy
        if labels is None:
            labels = torch.zeros(num_negative_examples + 1, 1)
            labels[0, 0] = 1
            labels = labels.repeat(batch_size, 1)
            labels = labels.to(logits.device)
        if config.loss_fn == "2H_BCE":
            # Concatenate logits from both heads
            labels = labels.view(-1).long()
            logits = torch.cat((logits, logits_secondary), dim=1)
    elif config.loss_fn == "negative_sampling":
        logits = logits.view(-1, num_negative_examples + 1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    else:
        raise ValueError(f"Unknown loss function {config.loss_fn}")

    return logits, labels