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


class CrossEncoder(nn.Module):
    base_model_prefix = "reranker"

    def __init__(self, config):
        super().__init__()
        # Initialize the BERT model with a pooling layer
        self.bert_model = AttentionFusionBertModel(config, add_pooling_layer=True)
        # Define a classifier layer which projects the CLS token's embedding
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Define a sigmoid activation function to output a probability score
        self.sigmoid = nn.Sigmoid()

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

        # Pass the CLS token's output through the classifier to get the logit
        logits = self.classifier(cls_output)

        return logits

    def predict(self, input_embeds, attention_mask=None, token_type_ids=None):
        logits = self.forward(input_embeds, attention_mask, token_type_ids)
        probabilities = self.sigmoid(logits)

        return probabilities


class RerankModel(pl.LightningModule):
    """
    Class for RAG, re-implementation
    """

    def __init__(self, config: EasyDict, prepared_data) -> None:
        super().__init__()

        self.config = config
        self.prepared_data = prepared_data
        self.tokenizers = self.prepared_data["tokenizers"]
        self.tokenizer = self.tokenizers["tokenizer"]
        self.decoder_tokenizer = self.tokenizers["decoder_tokenizer"]
        self.init_model_base()
        self.init_reranker()
        self.loss_fn = nn.BCEWithLogitsLoss()

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

    #     else:
    #         self.context_text_encoder = FLMRTextModel(pretrain_config.text_config)
    #         self.context_text_encoder_linear = nn.Linear(pretrain_config.text_config.hidden_size, pretrain_config.dim, bias=False)
    #         self.context_vision_encoder = FLMRVisionModel(pretrain_config.vision_config)
    #         self.context_vision_projection = FLMRMultiLayerPerceptron(
    #             (
    #                 self.vision_encoder_embedding_size,
    #                 (self.late_interaction_embedding_size * self.mapping_network_prefix_length) // 2,
    #                 self.late_interaction_embedding_size * self.mapping_network_prefix_length,
    #             )
    #         )
    #         self.init_transformer_mapping()

    # def init_transformer_mapping(self):
    #     transformer_mapping_config_base = self.config.transformer_mapping_config_base
    #     transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)

    #     assert (
    #         self.config.text_config.hidden_size == transformer_mapping_config.hidden_size
    #     ), f"hidden_size {self.config.text_config.hidden_size} != transformer_mapping_config.hidden_size {transformer_mapping_config.hidden_size}. To use cross attention, the dimensions must match."
    #     # shallow transformer
    #     transformer_mapping_config.num_hidden_layers = self.config.transformer_mapping_num_hidden_layers
    #     # add cross attention
    #     transformer_mapping_config.is_decoder = True
    #     transformer_mapping_config.add_cross_attention = True

    #     # The linear layer from vision encoder to transformer input
    #     self.transformer_mapping_input_linear = nn.Linear(
    #         self.vision_encoder_embedding_size, transformer_mapping_config.hidden_size
    #     )

    #     # The transformer encoder
    #     self.transformer_mapping_network = BertEncoder(transformer_mapping_config)

    #     # The linear layer from transformer output to FLMR dim
    #     self.transformer_mapping_output_linear = nn.Linear(
    #         transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
    #     )

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_pixel_values: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        num_negative_examples: int,
        preflmr_scores: Optional[torch.Tensor] = None,
    ):

        batch_size = query_input_ids.shape[0]
        expanded_batch_size = batch_size * (num_negative_examples + 1)
        assert expanded_batch_size == context_input_ids.shape[0]
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

        # # Prune the input size when the appended documents are too long
        # if joint_query_input_ids.size(1) > self.max_position_embeddings:
        #     joint_query_input_ids = joint_query_input_ids[:, :self.max_position_embeddings]
        #     joint_query_attention_mask = joint_query_attention_mask[:, :self.max_position_embeddings]

        query_outputs = self.query(
            joint_query_input_ids,
            joint_query_attention_mask,
            query_pixel_values,
            None,
            None,
            None,
        )
        reranker_inputs = self.cross_encoder_input_mapping(
            query_outputs.late_interaction_output
        )

        image_token_size = reranker_inputs.size(1) - joint_query_attention_mask.size(1)

        # All vision embeddings should be used in the attention
        expand_mask = torch.ones(
            expanded_batch_size,
            image_token_size,
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
                query_text_size + image_token_size,
            )
            
            # Query Self-Attention Mask
            upper_left = torch.zeros(
                (
                    expanded_batch_size,
                    query_text_size + image_token_size,
                    query_text_size + image_token_size,
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
            upper_right = truncated_scores.permute(0, 2, 1)
            bottom_left = truncated_scores
            
            
            
            
            reranker_attention_adj = torch.cat(
                [
                    torch.cat([upper_left, truncated_scores.permute(0, 2, 1)], dim=2),
                    torch.cat([truncated_scores, bottom_right], dim=2),
                ],
                dim=1,
            )
        else:
            reranker_attention_adj = None

        logits = self.reranker(
            reranker_inputs,
            attention_mask=reranker_attention_mask,
            attention_adj=reranker_attention_adj,
        )

        # First document is the positive example, concatenate them all along the first dimension and use binary cross entropy
        labels = torch.zeros(num_negative_examples + 1, 1)
        labels[0, 0] = 1
        labels = labels.repeat(batch_size, 1)
        labels = labels.to(logits.device)

        loss = self.loss_fn(logits, labels)
        return EasyDict(loss=loss, logits=logits)

    def query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Returns:

        """

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
            text_encoder_outputs = self.context_text_encoder(
                input_ids, attention_mask=attention_mask
            )
            text_encoder_hidden_states = text_encoder_outputs[
                0
            ]  # torch.Size([80, 512, 768])
            text_embeddings = self.context_text_encoder_linear(
                text_encoder_hidden_states
            )  # torch.Size([80, 512, 128])
            mask = (
                torch.tensor(
                    self.query_mask(input_ids, skiplist=[]), device=self.device
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
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_mask.squeeze(-1)
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

        Q = torch.cat([text_embeddings, vision_embeddings], dim=1)

        vision_encoder_attentions = (
            vision_encoder_outputs.attentions
            if vision_encoder_outputs is not None
            and hasattr(vision_encoder_outputs, "attentions")
            and output_attentions
            else None
        )
        vision_encoder_hidden_states = (
            vision_encoder_outputs.hidden_states
            if vision_encoder_outputs is not None
            and hasattr(vision_encoder_outputs, "hidden_states")
            and output_hidden_states
            else None
        )
        text_encoder_attentions = (
            text_encoder_outputs.attentions
            if text_encoder_outputs is not None
            and hasattr(text_encoder_outputs, "attentions")
            and output_attentions
            else None
        )
        text_encoder_hidden_states = (
            text_encoder_outputs.hidden_states
            if text_encoder_outputs is not None
            and hasattr(text_encoder_outputs, "hidden_states")
            and output_hidden_states
            else None
        )
        transformer_mapping_network_attentions = (
            transformer_mapping_outputs.attentions
            if transformer_mapping_outputs is not None
            and hasattr(transformer_mapping_outputs, "attentions")
            and output_attentions
            else None
        )
        transformer_mapping_network_hidden_states = (
            transformer_mapping_outputs.hidden_states
            if transformer_mapping_outputs is not None
            and hasattr(transformer_mapping_outputs, "hidden_states")
            and output_hidden_states
            else None
        )

        return FLMRQueryEncoderOutput(
            pooler_output=Q[:, 0, :],
            late_interaction_output=torch.nn.functional.normalize(Q, p=2, dim=2),
            vision_encoder_attentions=vision_encoder_attentions,
            vision_encoder_hidden_states=vision_encoder_hidden_states,
            text_encoder_attentions=text_encoder_attentions,
            text_encoder_hidden_states=text_encoder_hidden_states,
            transformer_mapping_network_attentions=transformer_mapping_network_attentions,
            transformer_mapping_network_hidden_states=transformer_mapping_network_hidden_states,
        )

    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
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

    def invert_attention_mask(self, encoder_attention_mask):
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
            dtype=self.dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask
        ) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask
