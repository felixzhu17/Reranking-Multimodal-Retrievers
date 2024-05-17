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
    if config.loss_fn in ["binary_cross_entropy", "two_head_binary_cross_entropy"]:
        pos_weight = torch.tensor([config.pos_weight], device=device) if config.pos_weight is not None else None
        if pos_weight is not None:
            print("Weighted BCE Loss", config.pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif config.loss_fn == "negative_sampling":
        print("Negative Sampling Loss")
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function {config.loss_fn}")
    return loss_fn

def prepare_logits_labels(config, logits, logits_secondary, batch_size, num_negative_examples):
    if config.loss_fn in ["binary_cross_entropy", "two_head_binary_cross_entropy"]:
        # First document is the positive example, concatenate them all along the first dimension and use binary cross entropy
        labels = torch.zeros(num_negative_examples + 1, 1)
        labels[0, 0] = 1
        labels = labels.repeat(batch_size, 1)
        labels = labels.to(logits.device)
        if config.loss_fn == "two_head_binary_cross_entropy":
            # Concatenate logits from both heads
            logits = torch.cat((logits, logits_secondary), dim=1)
            # Apply softmax across the concatenated logits
            logits = F.softmax(logits, dim=1)
            # Use only the probability of the positive class (assume positive class is the first one)
            logits = logits[:, 0].unsqueeze(1)
    elif config.loss_fn == "negative_sampling":
        logits = logits.view(-1, num_negative_examples + 1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    else:
        raise ValueError(f"Unknown loss function {config.loss_fn}")
    return logits, labels