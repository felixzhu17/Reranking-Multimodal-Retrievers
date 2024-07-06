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
from src.models.rerank.utils import initialise_loss_fn, prepare_logits_labels, prepare_full_context_inputs, CrossEncoder, invert_attention_mask
from src.models.rerank.mores_model import MORESSym

HEAD_TOKEN_LEEWAY = 4
class RerankModel(pl.LightningModule):
    """
    Class for RAG, re-implementation
    """

    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.config = config
        self.init_model_base()
        self.init_reranker()
        self.loss_fn = initialise_loss_fn(config, self.device)

    def init_reranker(self):
        cross_encoder_config_base = self.config.cross_encoder_config_base
        cross_encoder_config = BertConfig.from_pretrained(cross_encoder_config_base)
        cross_encoder_config.num_hidden_layers = (
            self.config.cross_encoder_num_hidden_layers
        )
        cross_encoder_config.max_position_embeddings = (
            self.config.cross_encoder_max_position_embeddings
        )
        self.reranker = CrossEncoder(cross_encoder_config)
        self.cross_encoder_input_mapping = nn.Linear(
            self.late_interaction_embedding_size, cross_encoder_config.hidden_size
        )

    def init_model_base(self):

        pretrain_config = FLMRConfig.from_pretrained(
            self.config.pretrain_model_version, trust_remote_code=True
        )
        self.late_interaction_embedding_size = pretrain_config.dim
        self.max_position_embeddings = (
            pretrain_config.text_config.max_position_embeddings
        )
        self.transformer_mapping_cross_attention_length = (
            pretrain_config.transformer_mapping_cross_attention_length
        )

        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
            self.config.pretrain_model_version, subfolder="query_tokenizer"
        )
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
            self.config.pretrain_model_version, subfolder="context_tokenizer"
        )

        pretrain_model = FLMRModelForRetrieval.from_pretrained(
            self.config.pretrain_model_version,
            config=pretrain_config,
            query_tokenizer=self.query_tokenizer,
            context_tokenizer=self.context_tokenizer,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.context_text_encoder = pretrain_model.context_text_encoder
        self.context_text_encoder_linear = pretrain_model.context_text_encoder_linear
        self.context_vision_encoder = pretrain_model.context_vision_encoder
        self.context_vision_projection = pretrain_model.context_vision_projection
        self.transformer_mapping_input_linear = (
            pretrain_model.transformer_mapping_input_linear
        )
        self.transformer_mapping_network = pretrain_model.transformer_mapping_network
        self.transformer_mapping_output_linear = (
            pretrain_model.transformer_mapping_output_linear
        )

        if pretrain_config.load_cpu_extension:
            try:
                FLMRModelForRetrieval.try_load_torch_extensions()
            except Exception as e:
                raise ValueError(
                    f"Unable to load `segmented_maxsim.cpp`. hf-hub does not download this file automatically. Please download it manually from `https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/blob/main/segmented_maxsim.cpp` and put it under the same folder as the model file.\n {e}"
                )

        if pretrain_config.mask_punctuation:
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [
                    symbol,
                    self.context_tokenizer.encode(symbol, add_special_tokens=False)[0],
                ]
            }

        if pretrain_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                pretrain_config.mask_instruction_token, add_special_tokens=False
            )[0]
            
        else:
            self.mask_instruction = False

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_pixel_values: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        num_negative_examples: int,
        preflmr_scores: Optional[torch.Tensor] = None,
        fusion_multiplier: float = 1,
        labels: Optional[List[int]] = None,
    ):

        if query_pixel_values is None:
            raise NotImplementedError("text_only is not implemented for this model")
        batch_size = query_input_ids.size(0)
        expanded_batch_size = batch_size * (num_negative_examples + 1)
        assert expanded_batch_size == context_input_ids.size(0)
        if labels:
            assert len(labels) == expanded_batch_size
        query_input_ids = query_input_ids.repeat_interleave(
            num_negative_examples + 1, dim=0
        ).contiguous()
        query_attention_mask = query_attention_mask.repeat_interleave(
            num_negative_examples + 1, dim=0
        ).contiguous()
        query_pixel_values = query_pixel_values.repeat_interleave(
            num_negative_examples + 1, dim=0
        ).contiguous()
        query_text_size = query_input_ids.size(1)
        context_text_size = context_input_ids.size(1)
        assert context_text_size == self.max_position_embeddings

        left_truncate_context_size = 2
        right_truncate_context_size = left_truncate_context_size - query_text_size

        joint_query_input_ids = torch.cat(
            [
                query_input_ids,
                context_input_ids[
                    :, left_truncate_context_size:right_truncate_context_size
                ],
            ],
            dim=1,
        )
        joint_query_attention_mask = torch.cat(
            [
                query_attention_mask,
                context_attention_mask[
                    :, left_truncate_context_size:right_truncate_context_size
                ],
            ],
            dim=1,
        )


        query_outputs = self.query(
            input_ids = joint_query_input_ids,
            attention_mask = joint_query_attention_mask,
            pixel_values = query_pixel_values,
            image_features = None,
            output_attentions = None,
            output_hidden_states = None,
            mask_instructions = self.mask_instruction,
            token_type_ids = None,
        )
        reranker_inputs = self.cross_encoder_input_mapping(
            query_outputs.late_interaction_output
        )
        
        joint_query_attention_mask = query_outputs.query_mask.squeeze(dim=-1)
        query_image_size = reranker_inputs.size(1) - joint_query_attention_mask.size(1)
    
        # Include vision embeddings so they are used in the attention mask
        expand_mask = torch.ones(
            expanded_batch_size,
            query_image_size,
            dtype=joint_query_attention_mask.dtype,
            device=joint_query_attention_mask.device,
        )
        
        reranker_attention_mask = torch.cat(
            [joint_query_attention_mask, expand_mask], dim=1
        )  # torch.Size([80, 593])

        # Reorder to Query, Image, Context
        reranker_inputs = torch.cat(
            (
                reranker_inputs[:, :query_text_size],
                reranker_inputs[:, context_text_size:],
                reranker_inputs[:, query_text_size:context_text_size],
            ),
            dim=1,
        )
    
        
        reranker_attention_mask = torch.cat(
            (
                reranker_attention_mask[:, :query_text_size],
                reranker_attention_mask[:, context_text_size:],
                reranker_attention_mask[:, query_text_size:context_text_size],
            ),
            dim=1,
        )

        if preflmr_scores is not None:
            truncated_scores = preflmr_scores[
                :, left_truncate_context_size:right_truncate_context_size, :
            ]
            assert truncated_scores.shape == (
                expanded_batch_size,
                context_text_size - query_text_size,
                query_text_size + query_image_size,
            )
            
            # Query Self-Attention Mask
            upper_left = torch.zeros(
                (
                    expanded_batch_size,
                    query_text_size + query_image_size,
                    query_text_size + query_image_size,
                ), device = self.device
            )
            
            # Context Self-Attention Mask
            bottom_right = torch.zeros(
                (
                    expanded_batch_size,
                    context_text_size - query_text_size,
                    context_text_size - query_text_size,
                ), device = self.device
            )
            
            # Cross-Attention Fusion
            upper_right = F.softmax(truncated_scores.permute(0, 2, 1), dim=-1)
            bottom_left = F.softmax(truncated_scores, dim=-1)
            
      
            reranker_attention_adj = torch.cat(
                [
                    torch.cat([upper_left, upper_right], dim=2),
                    torch.cat([bottom_left, bottom_right], dim=2),
                ],
                dim=1,
            ) * fusion_multiplier

            
        else:
            reranker_attention_adj = None

        logits, logits_secondary = self.reranker(
            reranker_inputs,
            attention_mask=reranker_attention_mask,
            attention_adj=reranker_attention_adj,
        )

        logits, labels = prepare_logits_labels(self.config, logits, logits_secondary, batch_size, num_negative_examples, labels = labels)
        loss = self.loss_fn(logits, logits)
        if self.config.loss_fn == "2H_BCE":
            logits = logits[:, 1].unsqueeze(1)
        return EasyDict(loss=loss, logits=logits)

    def query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        mask_instructions: bool = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
      

        input_modality = []
        if pixel_values is not None or image_features is not None:
            input_modality.append("image")
        if input_ids is not None and attention_mask is not None:
            input_modality.append("text")

        text_encoder_outputs = None
        vision_encoder_outputs = None
        transformer_mapping_outputs = None

        if "image" in input_modality:
            assert (
                pixel_values is not None or image_features is not None
            ), "pixel_values or image_features must be provided if image modality is used"
            assert (
                pixel_values is None or image_features is None
            ), "pixel_values and image_features cannot be provided at the same time"

        if "text" in input_modality:
            assert (
                input_ids is not None and attention_mask is not None
            ), "input_ids and attention_mask must be provided if text modality is used"
            # Forward the text encoder
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(
                self.device
            )
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            text_encoder_outputs = self.context_text_encoder(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            text_encoder_hidden_states = text_encoder_outputs[
                0
            ]  # torch.Size([80, 512, 768])
            text_embeddings = self.context_text_encoder_linear(
                text_encoder_hidden_states
            )  # torch.Size([80, 512, 128])
            
        
            mask = (
                torch.tensor(
                    self.query_mask(input_ids, skiplist=[], mask_instructions = mask_instructions), device=self.device
                )
                .unsqueeze(2)
                .float()
            )  # torch.Size([80, 512, 1])
            text_embeddings = text_embeddings * mask
       
        if "image" in input_modality:
            if pixel_values is not None:
                batch_size = pixel_values.shape[0]
                # Forward the vision encoder
                pixel_values = pixel_values.to(self.device)
                if len(pixel_values.shape) == 5:
                    # Multiple ROIs are provided
                    # merge the first two dimensions
                    pixel_values = pixel_values.reshape(
                        -1,
                        pixel_values.shape[2],
                        pixel_values.shape[3],
                        pixel_values.shape[4],
                    )
                vision_encoder_outputs = self.context_vision_encoder(
                    pixel_values, output_hidden_states=True
                )
                vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]

            if image_features is not None:
                batch_size = image_features.shape[0]
                vision_embeddings = image_features.to(self.device)

            # Forward the vision projection / mapping network
            vision_embeddings = self.context_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(
                batch_size, -1, self.late_interaction_embedding_size
            )

            # select the second last layer
            vision_second_last_layer_hidden_states = (
                vision_encoder_outputs.hidden_states[-2][:, 1:]
            )
            # transformer_mapping
            transformer_mapping_input_features = self.transformer_mapping_input_linear(
                vision_second_last_layer_hidden_states
            )

            # Cross attention only attends to the first 32 tokens
            encoder_mask = torch.ones_like(mask).to(
                mask.device, dtype=mask.dtype
            )  # torch.Size([80, 512, 1])

            cross_attention_length = self.transformer_mapping_cross_attention_length
            if text_encoder_hidden_states.shape[1] > cross_attention_length:
                text_encoder_hidden_states = text_encoder_hidden_states[
                    :, :cross_attention_length
                ]
                encoder_mask = encoder_mask[:, :cross_attention_length]

            # Obtain cross attention mask
            encoder_extended_attention_mask = invert_attention_mask(
                encoder_mask.squeeze(-1), self.dtype
            )  # torch.Size([80, 1, 1, 32])

            # Pass through the transformer mapping
            transformer_mapping_outputs = self.transformer_mapping_network(
                transformer_mapping_input_features,  # torch.Size([80, 49, 768])
                encoder_hidden_states=text_encoder_hidden_states,  # torch.Size([80, 32, 768])
                encoder_attention_mask=encoder_extended_attention_mask,  # torch.Size([80, 1, 1, 32])
            )

            transformer_mapping_output_features = (
                transformer_mapping_outputs.last_hidden_state
            )
            
            # Convert the dimension to FLMR dim
            transformer_mapping_output_features = (
                self.transformer_mapping_output_linear(
                    transformer_mapping_output_features
                )
            )

            vision_embeddings = torch.cat(
                [vision_embeddings, transformer_mapping_output_features], dim=1
            )  # 32, 49, torch.Size([80, 81, 128])

        if "image" in input_modality:
            Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        else:
            Q = text_embeddings
        

        return EasyDict(pooler_output = Q[:, 0, :], 
                        late_interaction_output = torch.nn.functional.normalize(Q, p=2, dim=2),
                        query_mask = mask,)

    def query_mask(self, input_ids, skiplist, mask_instructions = None):        
        mask_instructions = mask_instructions if mask_instructions is not None else self.mask_instruction
        if not mask_instructions:
            return self.mask(input_ids, skiplist)

        # find the position of end of instruction in input_ids
        # mask the tokens before the position
        sep_id = self.instruction_token_id
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        # if any of the positions is lower than 1, set to 1
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
                logger.error(
                    f"can not find the separator in the input_ids: {input_ids[i].tolist()}"
                )
        mask = [
            [
                (x not in skiplist)
                and (x != 0)
                and (index > sep_positions[seq_index] or index < 2)
                for index, x in enumerate(d)
            ]
            for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        return mask
    
    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != 0) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask

class FullContextRerankModel(RerankModel):
    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)
        self.max_query_length = self.config.max_query_length
        self.max_decoder_source_length = self.config.max_decoder_source_length
        self.max_context_length = self.max_decoder_source_length - self.max_query_length - HEAD_TOKEN_LEEWAY
        
    
    def forward(self, query_text_sequences, query_pixel_values, context_text_sequences, num_negative_examples, labels = None):
        text_only = query_pixel_values is None
        batch_size = len(query_text_sequences)
        expanded_batch_size = batch_size * (num_negative_examples + 1)
        assert expanded_batch_size == len(context_text_sequences)
        if labels:
            assert len(labels) == expanded_batch_size
        
        inputs = prepare_full_context_inputs(
            query_text_sequences, 
            context_text_sequences, 
            self.query_tokenizer, 
            self.max_query_length, 
            self.max_context_length, 
            self.max_decoder_source_length, 
            num_negative_examples + 1
        )
        
        if not text_only:
            query_pixel_values = query_pixel_values.repeat_interleave(
                num_negative_examples + 1, dim=0
            ).contiguous()
        
        query_outputs = self.query(
            input_ids = inputs.input_ids,
            attention_mask = inputs.attention_mask,
            pixel_values = query_pixel_values,
            image_features = None,
            output_attentions = None,
            output_hidden_states = None,
            mask_instructions = False,
            token_type_ids = inputs.token_type_ids,
        )
        
        reranker_inputs = self.cross_encoder_input_mapping(
            query_outputs.late_interaction_output
        )
        
        if text_only:
            reranker_attention_mask = query_outputs.query_mask.squeeze(dim=-1)
        else:
            joint_query_attention_mask = query_outputs.query_mask.squeeze(dim=-1)
            query_image_size = reranker_inputs.size(1) - joint_query_attention_mask.size(1)
        
            # Include vision embeddings so they are used in the attention mask
            expand_mask = torch.ones(
                expanded_batch_size,
                query_image_size,
                dtype=joint_query_attention_mask.dtype,
                device=joint_query_attention_mask.device,
            )
            
            reranker_attention_mask = torch.cat(
                [joint_query_attention_mask, expand_mask], dim=1
            )  # torch.Size([80, 593])


        logits, logits_secondary = self.reranker(
            reranker_inputs,
            attention_mask=reranker_attention_mask,
        )
        logits, labels = prepare_logits_labels(self.config, logits, logits_secondary, batch_size, num_negative_examples, labels = labels)
        
        #MODIFIED
        loss = self.loss_fn(logits, labels)
        # loss = F.binary_cross_entropy_with_logits(logits, logits, reduction='none')
        if self.config.loss_fn == "2H_BCE":
            logits = logits[:, 1].unsqueeze(1)
        return EasyDict(loss=loss, logits=logits)
    