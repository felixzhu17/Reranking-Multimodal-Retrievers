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
from src.models.rerank.utils import initialise_loss_fn, prepare_logits_labels
from transformers import Blip2ForConditionalGeneration, Blip2Config, Blip2Processor

POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"

class DecoderRerankModel(pl.LightningModule):
    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.config = config
        self.init_reranker()
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        
    def init_reranker(self):
        GeneratorModelClass = globals()[self.config.GeneratorModelClass]
        GeneratorConfigClass = globals()[self.config.GeneratorConfigClass]
        generator_model_config = GeneratorConfigClass.from_pretrained(
            self.config.GeneratorModelVersion
        )
        self.model = GeneratorModelClass.from_pretrained(
            self.config.GeneratorModelVersion,
            config=generator_model_config,
        )
        self.tokenizer = Blip2Processor.from_pretrained(self.config.GeneratorModelVersion)
        self.yes_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(POSITIVE_LABEL)
        self.no_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(NEGATIVE_LABEL)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.model = PeftModelForSeq2SeqLM(
            self.model, peft_config
        )
        self.model.print_trainable_parameters()

    def forward(self, query_text_sequences, query_pixel_values, context_text_sequences, num_negative_examples):
        docs_per_query = num_negative_examples + 1
        concatenated_sequences = [
            f"Query: {query_text_sequences[i]} Document: {context_text_sequences[i * docs_per_query + j]} Relevant:"
            for i in range(len(query_text_sequences))
            for j in range(docs_per_query)
        ]
        labels = [
            POSITIVE_LABEL if j == 0 else NEGATIVE_LABEL
            for _ in range(len(query_text_sequences))
            for j in range(docs_per_query)
        ]
                    
        inputs = self.tokenizer(text=concatenated_sequences, return_tensors="pt", padding=True, truncation=True)
        target_tokens = self.tokenizer(text=labels, return_tensors="pt", padding=True, truncation=True).input_ids
        labels = target_tokens.reshape(-1, 2) 
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        inputs['pixel_values'] = query_pixel_values.repeat_interleave(docs_per_query,0).to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits[:,0,:]
        
        # Extract logits for 'yes' and 'no' tokens
        yes_logits = logits[..., self.yes_token_id]
        no_logits = logits[..., self.no_token_id]
        stacked_logits = torch.stack([yes_logits, no_logits], dim=-1)
        probabilities = torch.nn.functional.softmax(stacked_logits, dim=-1)

        # Extract the probabilities for the 'yes' token
        yes_probabilities = probabilities[..., 0].unsqueeze(1)
        return EasyDict(loss=loss, logits=yes_probabilities)
    
    
class DecoderHeadRerankModel(pl.LightningModule):
    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.config = config
        self.init_reranker()
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        self.yes_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(POSITIVE_LABEL)
        self.no_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(NEGATIVE_LABEL)

    def init_reranker(self):
        GeneratorModelClass = globals()[self.config.GeneratorModelClass]
        GeneratorConfigClass = globals()[self.config.GeneratorConfigClass]
        generator_model_config = GeneratorConfigClass.from_pretrained(
            self.config.GeneratorModelVersion
        )
        self.model = GeneratorModelClass.from_pretrained(
            self.config.GeneratorModelVersion,
            config=generator_model_config,
        )
        self.tokenizer = Blip2Processor.from_pretrained(self.config.GeneratorModelVersion)


        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.model = PeftModelForSeq2SeqLM(
            self.model, peft_config
        )
        self.model.print_trainable_parameters()

    def forward(self, query_text_sequences, query_pixel_values, context_text_sequences, num_negative_examples):
        docs_per_query = num_negative_examples + 1
        concatenated_sequences = [
            f"Query: {query_text_sequences[i]} Document: {context_text_sequences[i * docs_per_query + j]} Relevant:"
            for i in range(len(query_text_sequences))
            for j in range(docs_per_query)
        ]
        labels = [
            POSITIVE_LABEL if j == 0 else NEGATIVE_LABEL
            for _ in range(len(query_text_sequences))
            for j in range(docs_per_query)
        ]
                    
        inputs = self.tokenizer(text=concatenated_sequences, return_tensors="pt", padding=True, truncation=True)
        target_tokens = self.tokenizer(text=labels, return_tensors="pt", padding=True, truncation=True).input_ids
        labels = target_tokens.reshape(-1, 2) 
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        inputs['pixel_values'] = query_pixel_values.repeat_interleave(docs_per_query,0).to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits[:,0,:]
        
        # Extract logits for 'yes' and 'no' tokens
        yes_logits = logits[..., self.yes_token_id]
        no_logits = logits[..., self.no_token_id]
        stacked_logits = torch.stack([yes_logits, no_logits], dim=-1)
        probabilities = torch.nn.functional.softmax(stacked_logits, dim=-1)

        # Extract the probabilities for the 'yes' token
        yes_probabilities = probabilities[..., 0].unsqueeze(1)
        return EasyDict(loss=loss, logits=yes_probabilities)