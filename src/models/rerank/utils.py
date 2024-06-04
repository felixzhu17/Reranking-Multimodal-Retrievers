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
HEAD_TOKEN_LEEWAY = 4
class CrossEncoder(nn.Module):
    base_model_prefix = "reranker"

    def __init__(self, config):
        super().__init__()
        # Initialize the BERT model with a pooling layer
        self.bert_model = AttentionFusionBertModel(config, add_pooling_layer=True)
        # Define two classifier layers which project the CLS token's embedding
        self.classifier1 = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 1)
        # Define a sigmoid activation function to output a probability score

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        attention_adj=None,
        token_type_ids=None,
    ):
        # Forward pass through BERT model
        outputs = self.bert_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_adj=attention_adj,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # Get the CLS token's output (first token of sequence output)
        cls_output = outputs.last_hidden_state[:, 0]

        # Pass the CLS token's output through the classifier to get the logits
        logits1 = self.classifier1(cls_output)
        logits2 = self.classifier2(cls_output)

        return logits1, logits2
PREFIXES = [
    "Using the provided image, obtain documents that address the subsequent question: ",
    "Retrieve documents that provide an answer to the question alongside the image: ",
    "Extract documents linked to the question provided in conjunction with the image: ",
    "Utilizing the given image, obtain documents that respond to the following question: ",
    "Using the given image, access documents that provide insights into the following question: ",
    "Obtain documents that correspond to the inquiry alongside the provided image: ",
    "With the provided image, gather documents that offer a solution to the question: ",
    "Utilizing the given image, obtain documents that respond to the following question: ",
]

def remove_prefixes(text):
    return [remove_prefix(s) for s in text]

def remove_prefix(text):
    for prefix in PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text

def prepare_full_context_inputs(query_text_sequences, context_text_sequences, tokenizer, max_query_length, max_context_length, max_decoder_source_length, docs_per_query):
    # Tokenize and truncate query sequences
    truncated_query = [
        tokenizer.decode(tokenizer.encode(
            input_text, add_special_tokens=False, 
            max_length=max_query_length, truncation=True
        )) for input_text in query_text_sequences
    ]

    # Tokenize and truncate context sequences
    truncated_context = [
        tokenizer.decode(tokenizer.encode(
            context_text, add_special_tokens=False, 
            max_length=max_context_length, truncation=True
        )) for context_text in context_text_sequences
    ]

    concatenated_sequences = []

    # Concatenate sequences using the provided prompt template function
    for i, input_text in enumerate(truncated_query):
        for j in range(docs_per_query):
            context_index = i * docs_per_query + j
            context_text = truncated_context[context_index]
            concatenated_sequences.append((input_text, context_text))


    # Process the concatenated sequences into the desired input format
    inputs = tokenizer.batch_encode_plus(
        concatenated_sequences, 
        add_special_tokens=True,
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=max_decoder_source_length,
        return_attention_mask=True
    )

    return inputs

def prepare_decoder_inputs(query_text_sequences, context_text_sequences, prompt_template_func, processor, max_query_length, max_context_length, max_decoder_source_length, docs_per_query):
    # Tokenize and truncate query sequences
    truncated_query = [
        processor.tokenizer.decode(processor.tokenizer.encode(
            f"Query: {input_text}", add_special_tokens=False, 
            max_length=max_query_length, truncation=True
        )) for input_text in query_text_sequences
    ]

    # Tokenize and truncate context sequences
    truncated_context = [
        processor.tokenizer.decode(processor.tokenizer.encode(
            f"Document: {context_text}", add_special_tokens=False, 
            max_length=max_context_length, truncation=True
        )) for context_text in context_text_sequences
    ]

    concatenated_sequences = []

    # Concatenate sequences using the provided prompt template function
    for i, input_text in enumerate(truncated_query):
        for j in range(docs_per_query):
            context_index = i * docs_per_query + j
            context_text = truncated_context[context_index]
            concatenated_sequence = prompt_template_func(input_text, context_text)
            concatenated_sequences.append(concatenated_sequence)

    # Process the concatenated sequences into the desired input format
    inputs = processor(
        text=concatenated_sequences, 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=max_decoder_source_length
    )

    return inputs
    
    
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

def invert_attention_mask(encoder_attention_mask, dtype):
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    encoder_extended_attention_mask = (
        1.0 - encoder_extended_attention_mask
    ) * torch.finfo(dtype).min

    return encoder_extended_attention_mask