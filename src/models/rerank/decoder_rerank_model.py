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
from src.models.rerank.utils import initialise_loss_fn, prepare_logits_labels, prepare_decoder_inputs
from transformers import Blip2ForConditionalGeneration, Blip2Config, Blip2Processor

POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"
GENERATION_TOKEN = "<GEN>"
HEAD_TOKEN_LEEWAY = 4


class DecoderRerankModel(pl.LightningModule):
    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.config = config
        self.init_reranker()
        self.prompt_template_func = lambda input_text, context_text: f"{input_text} {context_text} Relevant:"
        
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
        self.processor = Blip2Processor.from_pretrained(self.config.GeneratorModelVersion)
        self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids(POSITIVE_LABEL)
        self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids(NEGATIVE_LABEL)

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

        self.max_query_length = self.config.max_query_length
        self.max_decoder_source_length = self.config.max_decoder_source_length
        self.max_context_length = self.max_decoder_source_length - self.max_query_length - HEAD_TOKEN_LEEWAY


    def forward(self, query_text_sequences, query_pixel_values, context_text_sequences, num_negative_examples, labels = None):
        docs_per_query = num_negative_examples + 1

                    
        inputs = prepare_decoder_inputs(
            query_text_sequences, context_text_sequences, self.prompt_template_func, 
            self.processor, self.max_query_length, self.max_context_length, 
            self.max_decoder_source_length, docs_per_query
        )
        
        if labels is None:
            labels = [
                POSITIVE_LABEL if j == 0 else NEGATIVE_LABEL
                for _ in range(len(query_text_sequences))
                for j in range(docs_per_query)
            ]
        else:
            labels = [POSITIVE_LABEL if label == 1 else NEGATIVE_LABEL for label in labels]
        
        target_tokens = self.processor(text=labels, return_tensors="pt", padding="longest", truncation=True).input_ids
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
        self.loss_fn = initialise_loss_fn(config, self.device)
        self.prompt_template_func = lambda input_text, context_text: f"{input_text} {context_text} {GENERATION_TOKEN}"
        
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
        self.processor = Blip2Processor.from_pretrained(self.config.GeneratorModelVersion)
        self.processor.tokenizer.add_special_tokens({'additional_special_tokens': [GENERATION_TOKEN]})
        self.decoder_start_token_id = self.model.language_model.config.decoder_start_token_id
        
        self.gen_score_id = self.processor.tokenizer.convert_tokens_to_ids([GENERATION_TOKEN])[0]
        self.classifier1 = nn.Linear(generator_model_config.text_config.hidden_size, 1, bias=False).to(self.device)
        self.classifier2 = nn.Linear(generator_model_config.text_config.hidden_size, 1, bias=False).to(self.device)


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
        
        self.max_query_length = self.config.max_query_length
        self.max_decoder_source_length = self.config.max_decoder_source_length
        self.max_context_length = self.max_decoder_source_length - self.max_query_length - HEAD_TOKEN_LEEWAY


    def forward(self, query_text_sequences, query_pixel_values, context_text_sequences, num_negative_examples, labels = None):
        docs_per_query = num_negative_examples + 1
        batch_size = len(query_text_sequences)

        inputs = prepare_decoder_inputs(
            query_text_sequences, context_text_sequences, self.prompt_template_func, 
            self.processor, self.max_query_length, self.max_context_length, 
            self.max_decoder_source_length, docs_per_query
        )
                
        # input_ids = inputs.input_ids.to(self.device)
        # attention_mask = inputs.attention_mask.to(self.device)
        # pixel_values = query_pixel_values.repeat_interleave(docs_per_query,0).to(self.device)        
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        inputs['pixel_values'] = query_pixel_values.repeat_interleave(docs_per_query,0).to(self.device)

        if self.config.GeneratorModelVersion == "Salesforce/blip2-flan-t5-xl":
            decoder_input_ids = torch.full(
                (batch_size*docs_per_query, 1), 
                self.decoder_start_token_id,
                dtype=torch.long
            ).to(self.device)
            inputs['decoder_input_ids'] = decoder_input_ids

        outputs = self.model(**inputs,
                             output_hidden_states = True)

        if self.config.GeneratorModelVersion == "Salesforce/blip2-flan-t5-xl":
            rel_hidden_states = outputs.language_model_outputs.decoder_hidden_states[-1].squeeze(1)
        else:
            hidden_states = outputs.language_model_outputs.hidden_states[-1]
            rel_position = (torch.eq(inputs.input_ids, self.gen_score_id).long().argmax(-1)).to(hidden_states.device)
            rel_hidden_states = hidden_states[torch.arange(hidden_states.size()[0], device=hidden_states.device), rel_position]
        logits, logits_secondary = self.classifier1(rel_hidden_states), self.classifier2(rel_hidden_states)
        logits, labels = prepare_logits_labels(self.config, logits, logits_secondary, batch_size, num_negative_examples, labels = labels)


        loss = self.loss_fn(logits, labels)
        return EasyDict(loss=loss, logits=logits)
        
        
        
