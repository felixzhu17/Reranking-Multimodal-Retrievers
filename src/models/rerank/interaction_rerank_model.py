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
from src.models.rerank.utils import initialise_loss_fn, prepare_logits_labels, CrossEncoder
from src.models.rerank.mores_model import MORESSym

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from easydict import EasyDict
from transformers import BertConfig
from typing import Optional, List


LATE_INTERACTION_EMBEDDING_SIZE = 128

class InteractionRerankModel(pl.LightningModule):

    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.config = config
        self.init_reranker()
        self.loss_fn = initialise_loss_fn(config, self.device)

    def init_reranker(self):
        cross_encoder_config_base = self.config.cross_encoder_config_base
        cross_encoder_config = BertConfig.from_pretrained(cross_encoder_config_base)
        cross_encoder_config.num_hidden_layers = self.config.cross_encoder_num_hidden_layers
        cross_encoder_config.max_position_embeddings = self.config.cross_encoder_max_position_embeddings

        if self.config.interaction_type == "MORES":
            print("USING MORES")
            self.reranker = MORESSym(cross_encoder_config)
        else:
            self.reranker = CrossEncoder(cross_encoder_config)

        self.late_interaction_embedding_size = LATE_INTERACTION_EMBEDDING_SIZE
        self.cross_encoder_input_mapping = nn.Linear(self.late_interaction_embedding_size, cross_encoder_config.hidden_size)

    def forward(
        self,
        query_late_interaction: torch.Tensor,
        context_late_interaction: torch.Tensor,
        num_negative_examples: int,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
        preflmr_scores: Optional[torch.Tensor] = None,
        fusion_multiplier: float = 1,
        labels: Optional[List[int]] = None,
    ):
        batch_size = query_late_interaction.size(0)
        expanded_batch_size = batch_size * (num_negative_examples + 1)
        assert expanded_batch_size == context_late_interaction.size(0), f"{query_late_interaction.shape}, {context_late_interaction.shape}, {num_negative_examples}"

        query_length = query_late_interaction.size(1)
        context_length = context_late_interaction.size(1)

        query_late_interaction = query_late_interaction.repeat_interleave(num_negative_examples + 1, dim=0).contiguous()
        query_mask = query_mask.repeat_interleave(num_negative_examples + 1, dim=0).contiguous()

        if preflmr_scores is not None:
            upper_left = torch.zeros((expanded_batch_size, query_length, query_length), device=self.device)
            bottom_right = torch.zeros((expanded_batch_size, context_length, context_length), device=self.device)
            upper_right = F.softmax(preflmr_scores.permute(0, 2, 1), dim=-1)
            bottom_left = F.softmax(preflmr_scores, dim=-1)
            reranker_attention_adj = torch.cat(
                [
                    torch.cat([upper_left, upper_right], dim=2),
                    torch.cat([bottom_left, bottom_right], dim=2),
                ],
                dim=1,
            ) * fusion_multiplier
        else:
            reranker_attention_adj = None



        if self.config.interaction_type == "MORES":
            query_inputs = self.cross_encoder_input_mapping(query_late_interaction)
            context_inputs = self.cross_encoder_input_mapping(context_late_interaction.to(torch.float32))
            logits, logits_secondary = self.reranker(
                qry=query_inputs,
                doc=context_inputs,
                qry_mask=query_mask,
                cross_mask=context_mask,
                attention_adj=reranker_attention_adj,
            )
        else:
            reranker_inputs = torch.cat((query_late_interaction, context_late_interaction), dim=1)
            reranker_inputs = self.cross_encoder_input_mapping(reranker_inputs)
            reranker_attention_mask = torch.cat((query_mask, context_mask), dim=1)
            logits, logits_secondary = self.reranker(reranker_inputs, attention_mask=reranker_attention_mask, attention_adj=reranker_attention_adj)

        logits, labels = prepare_logits_labels(self.config, logits, logits_secondary, batch_size, num_negative_examples, labels=labels)
        loss = self.loss_fn(logits, labels)
        return EasyDict(loss=loss, logits=logits)