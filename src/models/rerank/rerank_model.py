

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
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
from transformers import VisualBertModel, VisualBertConfig, BertTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from transformers import Blip2ForConditionalGeneration, Blip2Config
from src.models.retriever.retriever_dpr import RetrieverDPR

# For ColBERT model
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.colbert import ColBERT
from src.models.retriever.visual_colbert import *
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.data import Queries
from colbert import Searcher

from transformers.models.rag.retrieval_rag import Index

import pytorch_lightning as pl

import time

import logging
logger = logging.getLogger(__name__)


import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
import faiss
import pickle
from typing import Iterable, List, Optional, Tuple
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import random
from src.models.custom_peft import PeftModelForSeq2SeqLM
from src.models.flmr.models.flmr.modeling_flmr import FLMRQueryEncoderOutput, FLMRTextModel, FLMRVisionModel, FLMRMultiLayerPerceptron
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import BertConfig


class CrossEncoder(PreTrainedModel):
    base_model_prefix = "reranker"

    def __init__(self, config):
        super().__init__(config)
        # Initialize the BERT model with a pooling layer
        self.bert_model = BertModel(config, add_pooling_layer=True)
        # Define a classifier layer which projects the CLS token's embedding
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Define a sigmoid activation function to output a probability score
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Forward pass through BERT model
        outputs = self.bert_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  return_dict=True)

        # Get the CLS token's output (first token of sequence output)
        cls_output = outputs.last_hidden_state[:, 0]

        # Pass the CLS token's output through the classifier to get the logit
        logits = self.classifier(cls_output)

        # Apply sigmoid activation to convert logits to probabilities
        probabilities = self.sigmoid(logits)

        return probabilities



class RerankModel(pl.LightningModule):
    '''
    Class for RAG, re-implementation
    '''
    def __init__(self, config: EasyDict, prepared_data) -> None:
        super().__init__()

        self.config = config
        self.prepared_data = prepared_data
        self.init_retrieve()
        self.init_model_base()
        self.init_reranker()
        self.loss_fn = nn.BCELoss()

    def init_retrieve(self):
        self.passage_id2doc = None #self.prepared_data['passages'].id2doc
        
        import json
        # load all predictions in 
        self.questionId2topPassages = {}
        for prediction_pkl in self.config.model_config.index_files.static_results:
            logger.info(f"Loading static retrieval results from {prediction_pkl}")
            if prediction_pkl.endswith('.json'):
                # load using json
                with open(prediction_pkl, 'r') as f:
                    predictions = json.load(f)['output']
                    for pred in predictions:
                        q_id = pred['question_id']
                        top_ranking_passages = pred['top_ranking_passages']
                        self.questionId2topPassages[q_id] = top_ranking_passages
            else:
                # Can use `src/tools/reduce_retrieval_result_file_size.py` to reduce json file size to speed up the loading
                # in this case, we load from a pkl file
                with open(prediction_pkl, 'rb') as f:
                    predictions = pickle.load(f)['output']
                    for pred in predictions:
                        q_id = pred['question_id']
                        top_ranking_passages = pred['top_ranking_passages']
                        self.questionId2topPassages[q_id] = top_ranking_passages
        logger.info(f"Loaded {len(self.questionId2topPassages)} static retrieval results.")

    def init_reranker(self):
        cross_encoder_config_base = self.config.cross_encoder_config_base
        cross_encoder_config = BertConfig.from_pretrained(cross_encoder_config_base)
        self.reranker = CrossEncoder(cross_encoder_config)

    def init_model_base(self):
        self.late_interaction_embedding_size = self.config.dim
        self.context_text_encoder = FLMRTextModel(self.config.text_config)
        self.context_text_encoder_linear = nn.Linear(self.config.text_config.hidden_size, self.config.dim, bias=False)
        self.context_vision_encoder = FLMRVisionModel(self.config.vision_config)
        self.context_vision_projection = FLMRMultiLayerPerceptron(
            (
                self.vision_encoder_embedding_size,
                (self.late_interaction_embedding_size * self.mapping_network_prefix_length) // 2,
                self.late_interaction_embedding_size * self.mapping_network_prefix_length,
            )
        )
        self.init_transformer_mapping()

    def init_transformer_mapping(self):
        transformer_mapping_config_base = self.config.transformer_mapping_config_base
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)

        assert (
            self.config.text_config.hidden_size == transformer_mapping_config.hidden_size
        ), f"hidden_size {self.config.text_config.hidden_size} != transformer_mapping_config.hidden_size {transformer_mapping_config.hidden_size}. To use cross attention, the dimensions must match."
        # shallow transformer
        transformer_mapping_config.num_hidden_layers = self.config.transformer_mapping_num_hidden_layers
        # add cross attention
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True

        # The linear layer from vision encoder to transformer input
        self.transformer_mapping_input_linear = nn.Linear(
            self.vision_encoder_embedding_size, transformer_mapping_config.hidden_size
        )

        # The transformer encoder
        self.transformer_mapping_network = BertEncoder(transformer_mapping_config)

        # The linear layer from transformer output to FLMR dim
        self.transformer_mapping_output_linear = nn.Linear(
            transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
        )

  
    def retrieve(self, 
                    input_ids: torch.Tensor,
                    question_ids: List, 
                    n_docs=None,
                    **kwargs):
        """A dummy retrieval function, retrieve from static results

        Args:
            input_ids (torch.Tensor): [description]
            attention_mask (torch.Tensor): [description]
            labels (torch.Tensor): [description]
            question_ids (List): [description]
            input_text_sequences (List): [description]
            n_docs ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if n_docs is None:
            n_docs = self.config.model_config.num_knowledge_passages
        
        n_docs_to_retrieve = self.config.model_config.num_knowledge_passages

        batch_size = input_ids.shape[0]

        pos_item_ids = kwargs.get('pos_item_ids', [None]*batch_size)
        if pos_item_ids is None:
            pos_item_ids = [None]*batch_size
        
        #####   Dummy Retrieval ####
        retrieved_docs = []
        doc_scores = []
        for question_id, pos_ids in zip(question_ids, pos_item_ids):
            annotation = self.questionId2topPassages.get(str(question_id), None)
            if annotation is None:
                annotation = [
                    {
                        'score': 10,
                        'title': '',
                        'content': '',
                        'passage_id': ''
                    }
                ]*n_docs
            
            if n_docs < n_docs_to_retrieve:
                # This helps to reduce the number of documents used in training so that model can fit in the GPU memory provided
                # randomly select n_docs from top n_docs_to_retrieve
                top_passages = random.sample(annotation[:n_docs_to_retrieve], n_docs)
            else:
                top_passages = annotation[:n_docs]
            
            if 'use_gt_docs_for_training' in self.config.model_config.modules and pos_ids is not None:
                annotation = []
                for i in range(n_docs):
                    annotation.append(
                        {
                            'score': 10,
                            'title': '',
                            'content': '',
                            'passage_id': random.sample(pos_ids, 1)[0]
                        }
                    )
                top_passages = annotation
            
            
            for p in top_passages:
                p['title'] = ''
                passage_id = p['passage_id']
                p['content'] = self.passage_id2doc.get(passage_id, "")

            retrieved_docs.append(top_passages)
            scores = [p['score'] for p in top_passages]
            doc_scores.append(scores)
        
        doc_scores = torch.FloatTensor(doc_scores).to(device=self.device)

        assert len(retrieved_docs) == batch_size

        return EasyDict(
            retrieved_docs=retrieved_docs,
            doc_scores=doc_scores,
        )

    def forward(self, query_input_ids: torch.Tensor,
                      query_pixel_values: torch.Tensor,
                      context_input_ids: torch.Tensor,
                      labels: torch.Tensor,
                      question_ids: List,
                      input_text_sequences: List,
                    **kwargs):

        batch_size = query_input_ids.shape[0]
        n_docs = self.config.model_config.num_knowledge_passages_in_training
        
        # Retrieve docs for given question inputs
        retrieval_results = self.retrieve(query_input_ids, question_ids, n_docs=n_docs)
        retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        query_input_ids = query_input_ids.repeat_interleave(n_docs + 1, dim=0).contiguous()
        query_pixel_values = query_pixel_values.repeat_interleave(n_docs + 1, dim=0).contiguous()
        query_text_ids = torch.cat([query_input_ids, context_input_ids], dim=0)
        query_outputs = self.query(query_input_ids, attention_mask, query_pixel_values, None, None, None)
        scores = self.reranker(query_outputs.late_interaction_output, attention_mask)
        labels = torch.zeros(1, n_docs + 1)
        labels[0, 0] = 1
        labels = labels.repeat(batch_size, 1)
        loss = self.loss_fn(scores, labels)
        return loss


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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

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
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            text_encoder_outputs = self.context_text_encoder(input_ids, attention_mask=attention_mask)
            text_encoder_hidden_states = text_encoder_outputs[0]
            text_embeddings = self.context_text_encoder_linear(text_encoder_hidden_states)
            mask = torch.tensor(self.query_mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()

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
                        -1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
                    )
                vision_encoder_outputs = self.context_vision_encoder(pixel_values, output_hidden_states=True)
                vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]

            if image_features is not None:
                batch_size = image_features.shape[0]
                vision_embeddings = image_features.to(self.device)

            # Forward the vision projection / mapping network
            vision_embeddings = self.context_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(batch_size, -1, self.late_interaction_embedding_size)

            if self.config.use_transformer_mapping_network:
                # select the second last layer
                vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[-2][:, 1:]
                # transformer_mapping
                transformer_mapping_input_features = self.transformer_mapping_input_linear(
                    vision_second_last_layer_hidden_states
                )

                # Cross attention only attends to the first 32 tokens
                encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
                cross_attention_length = self.config.transformer_mapping_cross_attention_length
                if text_encoder_hidden_states.shape[1] > cross_attention_length:
                    text_encoder_hidden_states = text_encoder_hidden_states[:, :cross_attention_length]
                    encoder_mask = encoder_mask[:, :cross_attention_length]

                # Obtain cross attention mask
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
                # Pass through the transformer mapping
                transformer_mapping_outputs = self.transformer_mapping_network(
                    transformer_mapping_input_features,
                    encoder_hidden_states=text_encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                )
                transformer_mapping_output_features = transformer_mapping_outputs.last_hidden_state
                # Convert the dimension to FLMR dim
                transformer_mapping_output_features = self.transformer_mapping_output_linear(
                    transformer_mapping_output_features
                )
                # Merge with the vision embeddings
                vision_embeddings = torch.cat([vision_embeddings, transformer_mapping_output_features], dim=1)

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
            if text_encoder_outputs is not None and hasattr(text_encoder_outputs, "attentions") and output_attentions
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



        


    def get_retrieval_labels(self, 
                            question_ids: List,
                            batch_answers: List, 
                            batch_retrieved_docs: List):
        
        def most_frequent(List):
            return max(set(List), key = List.count)

        retrieved_docs = batch_retrieved_docs
        log_result = {
            'recall': [],
            'precision': [],
            'gold_precision': [],
            'gold_recall': [],
        }
        labels = []
        selected_answers = []
        for question_id, answer_list, docs in zip(question_ids, batch_answers, retrieved_docs):
            
            filtered_answer_list = [ans for ans in answer_list if ans != '']
            gold_answer = most_frequent(filtered_answer_list)
            unique_answers = list(set(answer_list))
            counts = Counter(filtered_answer_list)
            answer_list_by_frequency = sorted(filtered_answer_list, key=lambda x: -counts[x])
            
            doc_texts = [doc['content'] for doc in docs]
            
            found_answers = []
            found_gold_answers = []

            
            if 'add_null_document' in self.config.model_config.modules:
                doc_texts = doc_texts[1:]

            this_batch_labels = [0] * len(doc_texts)
            K = len(doc_texts)

            def check_contain_entity(ans, doc_to_check):
                doc_id = doc_to_check['title']
                triplet = self.data_loader.data.fvqa_data.triplets.get(doc_id, None)
                if triplet is None:
                    logger.error(f'triplet id {doc_id} not found in the data!')
                    return False
                else:
                    triplet_entities = [triplet.e1_label.lower(), triplet.e2_label.lower()]
                    if ans in triplet_entities:
                        return True
                    else:
                        return False
            

            if 'use_entity_in_retrieval_labels' in self.config.model_config.modules:
                for index, passage_data in enumerate(docs):
                    for answer in unique_answers:
                        if check_contain_entity(answer.lower(), passage_data):
                            found_answers.append(answer)
                            this_batch_labels[index] = 1
                            break
                    if check_contain_entity(gold_answer.lower(), passage_data):
                        found_gold_answers.append(answer)
                        this_batch_labels[index] = 1

                for index, passage_data in enumerate(doc_texts):
                    # by default the gold answer is selected, regardless the existence of answer
                    selected_answer = gold_answer
                    # select answer that appears in the document and with highest frequency
                    if gold_answer.lower() in passage_data.lower():
                        pass # no change, by default the gold answer is selected
                    else:
                        for answer in answer_list_by_frequency:
                            if answer == gold_answer:
                                continue # not consider gold answer
                            if answer.lower() in passage_data.lower():
                                selected_answer = answer
                                break
                    selected_answers.append(selected_answer)

            elif 'use_triplet_in_retrieval_labels' in self.config.model_config.modules:
                item = self.data_loader.data.vqa_data.lookup.get(question_id, None)
                ref_triplet_ids = []
                for i in item.facts.values():
                    ref_triplet_ids.extend(i)
                
                for index, passage_data in enumerate(docs):
                    
                    if passage_data['title'] in ref_triplet_ids:
                        this_batch_labels[index] = 1
                        found_answers.append(passage_data['title'])
                        found_gold_answers.append(passage_data['title'])

                for index, passage_data in enumerate(doc_texts):
                    # by default the gold answer is selected, regardless the existence of answer
                    selected_answer = gold_answer
                    # select answer that appears in the document and with highest frequency
                    if gold_answer.lower() in passage_data.lower():
                        pass # no change, by default the gold answer is selected
                    else:
                        for answer in answer_list_by_frequency:
                            if answer == gold_answer:
                                continue # not consider gold answer
                            if answer.lower() in passage_data.lower():
                                selected_answer = answer
                                break
                    selected_answers.append(selected_answer)
                
            else:
                for index, passage_data in enumerate(doc_texts):
                    for answer in unique_answers:
                        if answer.lower() in passage_data.lower():
                            found_answers.append(answer)
                            this_batch_labels[index] = 1
                            break
                    if gold_answer.lower() in passage_data.lower():
                        found_gold_answers.append(answer)
                        this_batch_labels[index] = 1

                for index, passage_data in enumerate(doc_texts):
                    # by default the gold answer is selected, regardless the existence of answer
                    selected_answer = gold_answer
                    # select answer that appears in the document and with highest frequency
                    if gold_answer.lower() in passage_data.lower():
                        pass # no change, by default the gold answer is selected
                    else:
                        for answer in answer_list_by_frequency:
                            if answer == gold_answer:
                                continue # not consider gold answer
                            if answer.lower() in passage_data.lower():
                                selected_answer = answer
                                break
                    selected_answers.append(selected_answer)

            labels.append(this_batch_labels)
                    
            if len(found_answers) > 0:
                # At least one answer is retireved
                log_result['recall'].append(1)
            else:
                log_result['recall'].append(0)
            # The proportion of retrieved knowledge has an answer
            log_result['precision'].append(len(found_answers) / K)

            if len(found_gold_answers) > 0:
                # if gold answer is found
                log_result['gold_recall'].append(1)
            else:
                log_result['gold_recall'].append(0)
            # The proportion of retrieved knowledge has the gold answer
            log_result['gold_precision'].append(len(found_gold_answers) / K)

        labels = torch.FloatTensor(labels)
        return EasyDict(
            retrieval_labels=labels,
            selected_answers=selected_answers,
        )

    @staticmethod
    def DistanceCorrelation(tensor_1, tensor_2):
        # tensor_1, tensor_2: [channel]
        # ref: https://en.wikipedia.org/wiki/Distance_correlation
        channel = tensor_1.shape[0]
        zeros = torch.zeros(channel, channel).to(tensor_1.device)
        zero = torch.zeros(1).to(tensor_1.device)
        tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
        """cul distance matrix"""
        a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
        tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
        a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
        """cul distance correlation"""
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
        dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
        dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
        dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
        return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)





