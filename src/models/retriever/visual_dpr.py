import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from transformers import T5EncoderModel, T5Config
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from transformers import CLIPTextModel, CLIPTextConfig
from easydict import EasyDict

from .retriever_dpr import RetrieverDPR

import logging

logger = logging.getLogger(__name__)


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class VisualDPRForPretraining(RetrieverDPR):
    """
    Class of DPR with Vision Input
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_config = config.model_config

        self.mapping_network_prefix_length = (
            self.model_config.mapping_network_prefix_length
        )
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size

        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        if "freeze_dpr_doc_encoder" in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning(
                "freezing the DPR document encoder. If the query encoder is not separated, the query encoder is also frozen."
            )
            for name, param in self.item_encoder.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if "freeze_mapping_network" in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        image_features=None,
        labels=None,
        span_labels=None,
        **kwargs,
    ):
        # query encoder
        # query_outputs = self.query_encoder(input_ids=input_ids,
        #                                     attention_mask=attention_mask)
        # query_embeddings = query_outputs.pooler_output
        # if self.query_pooler is not None:
        #     query_embeddings = self.query_pooler(query_last_hidden_states)
        # query_embeddings = query_last_hidden_states
        # print('query_embeddings', query_embeddings.shape)

        # Get image features as query_embeddings
        image_features = image_features.to(self.device)
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]

        last_hidden_states = self.vision_projection(last_hidden_states)  # bz x 32*128

        last_hidden_states = last_hidden_states.view(
            batch_size, -1, self.lm_embedding_size
        )
        # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
        # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
        query_embeddings = last_hidden_states.sum(1)

        # item encoder
        item_outputs = self.item_encoder(
            input_ids=item_input_ids, attention_mask=item_attention_mask
        )
        item_embeddings = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_embeddings = self.item_pooler(item_last_hidden_states)
        # item_embeddings = item_last_hidden_states
        # print('item_embeddings', item_embeddings.shape)

        query_embeddings = query_embeddings.contiguous()
        item_embeddings = item_embeddings.contiguous()

        ################## in-batch negative sampling ###############
        if "negative_samples_across_gpus" in self.config.model_config.modules:
            # print("get rank", get_rank())
            # print("get world size", get_world_size())
            # Gather embeddings from other GPUs
            n_nodes = get_world_size()

            # Create placeholder to hold embeddings passed from other ranks
            global_query_embeddings_placeholder = [
                torch.zeros(*query_embeddings.shape).to(query_embeddings.device)
                for _ in range(n_nodes)
            ]
            global_item_embeddings_placeholder = [
                torch.zeros(*item_embeddings.shape).to(item_embeddings.device)
                for _ in range(n_nodes)
            ]
            dist.all_gather(
                global_query_embeddings_placeholder, query_embeddings.detach()
            )
            dist.all_gather(
                global_item_embeddings_placeholder, item_embeddings.detach()
            )

            global_query_embeddings = []
            global_item_embeddings = []
            # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
            # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
            # input()
            current_rank = get_rank()
            for rank_index, remote_q_embeddings in enumerate(
                global_query_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_query_embeddings.append(remote_q_embeddings)
                else:
                    global_query_embeddings.append(query_embeddings)

            for rank_index, remote_item_embeddings in enumerate(
                global_item_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_item_embeddings.append(remote_item_embeddings)
                else:
                    global_item_embeddings.append(item_embeddings)

            # Replace the previous variables with gathered tensors
            query_embeddings = torch.cat(global_query_embeddings)
            item_embeddings = torch.cat(global_item_embeddings)

        batch_size = query_embeddings.shape[0]
        batch_size_with_pos_and_neg = item_embeddings.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos

        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(
            labels.device
        )
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step * i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)

        in_batch_scores = torch.matmul(query_embeddings, item_embeddings.T)
        loss = self.loss_fn(in_batch_scores, in_batch_labels)

        return EasyDict(
            {
                "loss": loss,
            }
        )

    def generate_query_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        image_features=None,
    ):
        # query encoder
        # query_outputs = self.query_encoder(input_ids=input_ids,
        #                                     attention_mask=attention_mask)
        # query_last_hidden_states = query_outputs.pooler_output
        # if self.query_pooler is not None:
        #     query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        # query_embeddings = query_last_hidden_states

        # Get image features as query_embeddings
        image_features = image_features.to(self.device)
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]

        last_hidden_states = self.vision_projection(last_hidden_states)  # bz x 32*128

        last_hidden_states = last_hidden_states.view(
            batch_size, -1, self.lm_embedding_size
        )
        # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
        # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
        # print('last_hidden_states', last_hidden_states.shape)
        query_embeddings = last_hidden_states.sum(1)
        # print('query_embeddings', query_embeddings.shape)
        # input()

        return query_embeddings


class VisualDPRForRetrieval(RetrieverDPR):
    """
    Class of DPR with Vision Input
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_config = config.model_config

        self.mapping_network_prefix_length = (
            self.model_config.mapping_network_prefix_length
        )
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size

        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        if "freeze_dpr_doc_encoder" in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning(
                "freezing the DPR document encoder. If the query encoder is not separated, the query encoder is also frozen."
            )
            for name, param in self.item_encoder.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if "freeze_mapping_network" in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        image_features=None,
        labels=None,
        span_labels=None,
        **kwargs,
    ):
        # query encoder
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_embeddings = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_embeddings = self.query_pooler(query_embeddings)
        # query_embeddings = query_last_hidden_states

        # Get image features as query_embeddings
        image_features = image_features.to(self.device)
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]

        last_hidden_states = self.vision_projection(last_hidden_states)  # bz x 32*128

        last_hidden_states = last_hidden_states.view(
            batch_size, -1, self.lm_embedding_size
        )
        # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
        # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
        # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
        query_embeddings = query_embeddings + last_hidden_states.sum(1)

        # item encoder
        item_outputs = self.item_encoder(
            input_ids=item_input_ids, attention_mask=item_attention_mask
        )
        item_embeddings = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_embeddings = self.item_pooler(item_last_hidden_states)
        # item_embeddings = item_last_hidden_states
        # print('item_embeddings', item_embeddings.shape)

        query_embeddings = query_embeddings.contiguous()
        item_embeddings = item_embeddings.contiguous()

        ################## in-batch negative sampling ###############
        if "negative_samples_across_gpus" in self.config.model_config.modules:
            # print("get rank", get_rank())
            # print("get world size", get_world_size())
            # Gather embeddings from other GPUs
            n_nodes = get_world_size()

            # Create placeholder to hold embeddings passed from other ranks
            global_query_embeddings_placeholder = [
                torch.zeros(*query_embeddings.shape).to(query_embeddings.device)
                for _ in range(n_nodes)
            ]
            global_item_embeddings_placeholder = [
                torch.zeros(*item_embeddings.shape).to(item_embeddings.device)
                for _ in range(n_nodes)
            ]
            dist.all_gather(
                global_query_embeddings_placeholder, query_embeddings.detach()
            )
            dist.all_gather(
                global_item_embeddings_placeholder, item_embeddings.detach()
            )

            global_query_embeddings = []
            global_item_embeddings = []
            # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
            # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
            # input()
            current_rank = get_rank()
            for rank_index, remote_q_embeddings in enumerate(
                global_query_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_query_embeddings.append(remote_q_embeddings)
                else:
                    global_query_embeddings.append(query_embeddings)

            for rank_index, remote_item_embeddings in enumerate(
                global_item_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_item_embeddings.append(remote_item_embeddings)
                else:
                    global_item_embeddings.append(item_embeddings)

            # Replace the previous variables with gathered tensors
            query_embeddings = torch.cat(global_query_embeddings)
            item_embeddings = torch.cat(global_item_embeddings)

        batch_size = query_embeddings.shape[0]
        batch_size_with_pos_and_neg = item_embeddings.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos

        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(
            labels.device
        )
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step * i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)

        in_batch_scores = torch.matmul(query_embeddings, item_embeddings.T)
        loss = self.loss_fn(in_batch_scores, in_batch_labels)

        return EasyDict(
            {
                "loss": loss,
            }
        )

    def generate_query_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        image_features=None,
    ):
        # query encoder
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_last_hidden_states = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states

        # Get image features as query_embeddings
        image_features = image_features.to(self.device)
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]

        last_hidden_states = self.vision_projection(last_hidden_states)  # bz x 32*128

        last_hidden_states = last_hidden_states.view(
            batch_size, -1, self.lm_embedding_size
        )
        # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
        # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
        # print('last_hidden_states', last_hidden_states.shape)
        query_embeddings = query_embeddings + last_hidden_states.sum(1)
        # print('query_embeddings', query_embeddings.shape)
        # input()

        return query_embeddings


class VisualDPRWithMultiModalDocs(VisualDPRForRetrieval):
    """
    Class of DPR with Vision Input
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_config = config.model_config

        self.mapping_network_prefix_length = (
            self.model_config.mapping_network_prefix_length
        )
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size

        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        self.doc_vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        if "freeze_dpr_doc_encoder" in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning(
                "freezing the DPR document encoder. If the query encoder is not separated, the query encoder is also frozen."
            )
            for name, param in self.item_encoder.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if "freeze_mapping_network" in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if "freeze_doc_encoder_mapping_network" in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network in the document encoder.")
            for name, param in self.doc_vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        self.multimodal_docs = self.model_config.get("multimodal_docs", False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        image_features=None,
        item_image_features=None,
        labels=None,
        span_labels=None,
        **kwargs,
    ):
        # query encoder
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_embeddings = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_embeddings = self.query_pooler(query_embeddings)
        # query_embeddings = query_last_hidden_states

        # Get image features as query_embeddings
        if image_features is not None:
            image_features = image_features.to(self.device)
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]

            last_hidden_states = self.vision_projection(
                last_hidden_states
            )  # bz x 32*128

            last_hidden_states = last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
            query_embeddings = query_embeddings + last_hidden_states.sum(1)

        # item encoder
        item_outputs = self.item_encoder(
            input_ids=item_input_ids, attention_mask=item_attention_mask
        )
        item_embeddings = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_embeddings = self.item_pooler(item_embeddings)
        # item_embeddings = item_last_hidden_states

        if self.multimodal_docs:
            # Get item image features as item_embeddings
            item_image_features = item_image_features.to(self.device)
            item_last_hidden_states = item_image_features
            batch_size = item_last_hidden_states.shape[0]

            item_last_hidden_states = self.doc_vision_projection(
                item_last_hidden_states
            )  # bz x 32*128

            item_last_hidden_states = item_last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # print("item_last_hidden_states", item_last_hidden_states.shape)
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
            item_embeddings = item_embeddings + item_last_hidden_states.sum(1)

            # print('item_embeddings', item_embeddings.shape)

        query_embeddings = query_embeddings.contiguous()
        item_embeddings = item_embeddings.contiguous()

        ################## in-batch negative sampling ###############
        if "negative_samples_across_gpus" in self.config.model_config.modules:
            # print("get rank", get_rank())
            # print("get world size", get_world_size())
            # Gather embeddings from other GPUs
            n_nodes = get_world_size()

            # Create placeholder to hold embeddings passed from other ranks
            global_query_embeddings_placeholder = [
                torch.zeros(*query_embeddings.shape).to(query_embeddings.device)
                for _ in range(n_nodes)
            ]
            global_item_embeddings_placeholder = [
                torch.zeros(*item_embeddings.shape).to(item_embeddings.device)
                for _ in range(n_nodes)
            ]
            dist.all_gather(
                global_query_embeddings_placeholder, query_embeddings.detach()
            )
            dist.all_gather(
                global_item_embeddings_placeholder, item_embeddings.detach()
            )

            global_query_embeddings = []
            global_item_embeddings = []
            # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
            # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
            # input()
            current_rank = get_rank()
            for rank_index, remote_q_embeddings in enumerate(
                global_query_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_query_embeddings.append(remote_q_embeddings)
                else:
                    global_query_embeddings.append(query_embeddings)

            for rank_index, remote_item_embeddings in enumerate(
                global_item_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_item_embeddings.append(remote_item_embeddings)
                else:
                    global_item_embeddings.append(item_embeddings)

            # Replace the previous variables with gathered tensors
            query_embeddings = torch.cat(global_query_embeddings)
            item_embeddings = torch.cat(global_item_embeddings)

        batch_size = query_embeddings.shape[0]
        batch_size_with_pos_and_neg = item_embeddings.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos

        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(
            labels.device
        )
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step * i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)

        in_batch_scores = torch.matmul(query_embeddings, item_embeddings.T)
        loss = self.loss_fn(in_batch_scores, in_batch_labels)

        return EasyDict(
            {
                "loss": loss,
            }
        )

    def generate_query_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        image_features=None,
    ):
        # query encoder
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_last_hidden_states = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states

        if image_features is not None:
            # Get image features as query_embeddings
            image_features = image_features.to(self.device)
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]

            last_hidden_states = self.vision_projection(
                last_hidden_states
            )  # bz x 32*128

            last_hidden_states = last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # print('last_hidden_states', last_hidden_states.shape)
            query_embeddings = query_embeddings + last_hidden_states.sum(1)
            # print('query_embeddings', query_embeddings.shape)
            # input()

        return query_embeddings

    def generate_item_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        item_image_features=None,
    ):
        # item encoder
        item_outputs = self.item_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        item_last_hidden_states = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_last_hidden_states = self.item_pooler(item_last_hidden_states)
        item_embeddings = item_last_hidden_states

        if self.multimodal_docs:
            # Get item image features as item_embeddings
            item_image_features = item_image_features.to(self.device)
            item_last_hidden_states = item_image_features
            batch_size = item_last_hidden_states.shape[0]

            item_last_hidden_states = self.doc_vision_projection(
                item_last_hidden_states
            )  # bz x 6*768

            item_last_hidden_states = item_last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # print("item_last_hidden_states", item_last_hidden_states.shape)
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
            item_embeddings = item_embeddings + item_last_hidden_states.sum(1)

            # print('item_embeddings', item_embeddings.shape)

        return item_embeddings


class VisualDPRWithMultiModalDocsWithOnlyImages(VisualDPRForRetrieval):
    """
    Class of DPR with Vision Input
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_config = config.model_config

        self.mapping_network_prefix_length = (
            self.model_config.mapping_network_prefix_length
        )
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size

        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        self.doc_vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        if "freeze_dpr_doc_encoder" in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning(
                "freezing the DPR document encoder. If the query encoder is not separated, the query encoder is also frozen."
            )
            for name, param in self.item_encoder.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if "freeze_mapping_network" in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if "freeze_doc_encoder_mapping_network" in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network in the document encoder.")
            for name, param in self.doc_vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        self.multimodal_docs = self.model_config.get("multimodal_docs", False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        image_features=None,
        item_image_features=None,
        labels=None,
        span_labels=None,
        **kwargs,
    ):
        # query encoder
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_embeddings = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_embeddings = self.query_pooler(query_embeddings)
        # query_embeddings = query_last_hidden_states

        # Get image features as query_embeddings
        if image_features is not None:
            image_features = image_features.to(self.device)
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]

            last_hidden_states = self.vision_projection(
                last_hidden_states
            )  # bz x 32*128

            last_hidden_states = last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
            query_embeddings = query_embeddings + last_hidden_states.sum(1)

        # item encoder
        item_outputs = self.item_encoder(
            input_ids=item_input_ids, attention_mask=item_attention_mask
        )
        item_embeddings = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_embeddings = self.item_pooler(item_embeddings)
        # item_embeddings = item_last_hidden_states

        if self.multimodal_docs:
            # Get item image features as item_embeddings
            item_image_features = item_image_features.to(self.device)
            item_last_hidden_states = item_image_features
            batch_size = item_last_hidden_states.shape[0]

            item_last_hidden_states = self.doc_vision_projection(
                item_last_hidden_states
            )  # bz x 32*128

            item_last_hidden_states = item_last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # print("item_last_hidden_states", item_last_hidden_states.shape)
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
            item_embeddings = item_last_hidden_states.sum(1)

            # print('item_embeddings', item_embeddings.shape)

        query_embeddings = query_embeddings.contiguous()
        item_embeddings = item_embeddings.contiguous()

        ################## in-batch negative sampling ###############
        if "negative_samples_across_gpus" in self.config.model_config.modules:
            # print("get rank", get_rank())
            # print("get world size", get_world_size())
            # Gather embeddings from other GPUs
            n_nodes = get_world_size()

            # Create placeholder to hold embeddings passed from other ranks
            global_query_embeddings_placeholder = [
                torch.zeros(*query_embeddings.shape).to(query_embeddings.device)
                for _ in range(n_nodes)
            ]
            global_item_embeddings_placeholder = [
                torch.zeros(*item_embeddings.shape).to(item_embeddings.device)
                for _ in range(n_nodes)
            ]
            dist.all_gather(
                global_query_embeddings_placeholder, query_embeddings.detach()
            )
            dist.all_gather(
                global_item_embeddings_placeholder, item_embeddings.detach()
            )

            global_query_embeddings = []
            global_item_embeddings = []
            # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
            # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
            # input()
            current_rank = get_rank()
            for rank_index, remote_q_embeddings in enumerate(
                global_query_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_query_embeddings.append(remote_q_embeddings)
                else:
                    global_query_embeddings.append(query_embeddings)

            for rank_index, remote_item_embeddings in enumerate(
                global_item_embeddings_placeholder
            ):
                # We append the embeddings from other GPUs if this embedding does not require gradients
                if rank_index != current_rank:
                    global_item_embeddings.append(remote_item_embeddings)
                else:
                    global_item_embeddings.append(item_embeddings)

            # Replace the previous variables with gathered tensors
            query_embeddings = torch.cat(global_query_embeddings)
            item_embeddings = torch.cat(global_item_embeddings)

        batch_size = query_embeddings.shape[0]
        batch_size_with_pos_and_neg = item_embeddings.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos

        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(
            labels.device
        )
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step * i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)

        in_batch_scores = torch.matmul(query_embeddings, item_embeddings.T)
        loss = self.loss_fn(in_batch_scores, in_batch_labels)

        return EasyDict(
            {
                "loss": loss,
            }
        )

    def generate_query_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        image_features=None,
    ):
        # query encoder
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_last_hidden_states = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states

        if image_features is not None:
            # Get image features as query_embeddings
            image_features = image_features.to(self.device)
            last_hidden_states = image_features
            batch_size = last_hidden_states.shape[0]

            last_hidden_states = self.vision_projection(
                last_hidden_states
            )  # bz x 32*128

            last_hidden_states = last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # print('last_hidden_states', last_hidden_states.shape)
            query_embeddings = query_embeddings + last_hidden_states.sum(1)
            # print('query_embeddings', query_embeddings.shape)
            # input()

        return query_embeddings

    def generate_item_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        item_image_features=None,
    ):
        # item encoder
        item_outputs = self.item_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        item_last_hidden_states = item_outputs.pooler_output
        if self.item_pooler is not None:
            item_last_hidden_states = self.item_pooler(item_last_hidden_states)
        item_embeddings = item_last_hidden_states

        if self.multimodal_docs:
            # Get item image features as item_embeddings
            item_image_features = item_image_features.to(self.device)
            item_last_hidden_states = item_image_features
            batch_size = item_last_hidden_states.shape[0]

            item_last_hidden_states = self.doc_vision_projection(
                item_last_hidden_states
            )  # bz x 6*768

            item_last_hidden_states = item_last_hidden_states.view(
                batch_size, -1, self.lm_embedding_size
            )
            # print("item_last_hidden_states", item_last_hidden_states.shape)
            # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
            # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
            # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
            item_embeddings = item_last_hidden_states.sum(1)

            # print('item_embeddings', item_embeddings.shape)

        return item_embeddings


class VisualDPRForRAG(pl.LightningModule):
    """
    Class of DPR with Vision Input
    Used in RAG training - only the query encoder is used
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        QueryEncoderModelClass = DPRQuestionEncoder

        if "$" in self.config.model_config.QueryEncoderModelVersion:
            self.config.model_config.QueryEncoderModelVersion = os.path.join(
                self.config.root_exp_dir,
                self.config.model_config.QueryEncoderModelVersion.replace("$", ""),
            )

        QueryEncoderConfigClass = globals()[
            self.config.model_config.QueryEncoderConfigClass
        ]
        query_model_config = QueryEncoderConfigClass.from_pretrained(
            self.config.model_config.QueryEncoderModelVersion
        )
        # if query_model_config.model_type == 'clip_text_model':
        #     query_model_config.max_position_embeddings = 512
        self.query_encoder = QueryEncoderModelClass.from_pretrained(
            self.config.model_config.QueryEncoderModelVersion,
            config=query_model_config,
            ignore_mismatched_sizes=True,
        )

        self.model_config = config.model_config

        self.mapping_network_prefix_length = (
            self.model_config.mapping_network_prefix_length
        )
        self.vision_embedding_size = self.model_config.vision_embedding_size
        self.lm_embedding_size = self.model_config.lm_embedding_size

        self.vision_projection = MLP(
            (
                self.vision_embedding_size,
                (self.lm_embedding_size * self.mapping_network_prefix_length) // 2,
                self.lm_embedding_size * self.mapping_network_prefix_length,
            )
        )

        # Load vision mapping network from an existing checkpoint
        # TODO: save the mapping network separately in DPR testing

        if "$" in self.config.model_config.QueryEncoderVisionMappingPath:
            self.config.model_config.QueryEncoderVisionMappingPath = os.path.join(
                self.config.root_exp_dir,
                self.config.model_config.QueryEncoderVisionMappingPath.replace("$", ""),
            )
        checkpoint_to_load = self.config.model_config.QueryEncoderVisionMappingPath
        if not checkpoint_to_load or checkpoint_to_load == "":
            print("No checkpoint found.")
        else:
            # We manually load the state dict
            print(f"Loading from {checkpoint_to_load}")
            state_dict_from_ckpt = torch.load(
                checkpoint_to_load, map_location=self.device
            )["state_dict"]
            state_dict_from_model = self.state_dict()
            # Only load parameters with "vision_projection"
            state_dict_from_ckpt = {
                k.replace("model.vision_projection", "vision_projection"): v
                for k, v in state_dict_from_ckpt.items()
                if "vision_projection" in k
            }
            state_dict_from_model.update(state_dict_from_ckpt)
            self.load_state_dict(state_dict_from_model)
            print(
                f"Load the following parameters to vision_projection from the given checkpoint: {state_dict_from_ckpt.keys()}"
            )

        if "freeze_dpr_doc_encoder" in self.model_config.modules:
            # freeze the ColBERT model
            logger.warning(
                "freezing the DPR document encoder. If the query encoder is not separated, the query encoder is also frozen."
            )
            for name, param in self.item_encoder.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

        if "freeze_mapping_network" in self.model_config.modules:
            # freeze the ViT model
            logger.warning("freezing the mapping network.")
            for name, param in self.vision_projection.named_parameters():
                # print(f"freezed: {name}")
                param.requires_grad = False

    def resize_token_embeddings(self, dim, decoder_dim=None):
        self.query_encoder.resize_token_embeddings(dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        image_features=None,
        labels=None,
        **kwargs,
    ):
        # query encoder
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_embeddings = query_outputs.pooler_output
        # query_embeddings = query_last_hidden_states

        # Get image features as query_embeddings
        image_features = image_features.to(self.device)
        last_hidden_states = image_features
        batch_size = last_hidden_states.shape[0]

        last_hidden_states = self.vision_projection(last_hidden_states)  # bz x 32*128

        last_hidden_states = last_hidden_states.view(
            batch_size, -1, self.lm_embedding_size
        )
        # batch_size x prefix_len x lm_embedding_size --> batch_size x lm_embedding_size
        # mimics score addition: prefix_len image tokens, each token interacts with the item embedding
        # batch_size x lm_embeddding_size + batch_size x lm_embeddding_size
        query_embeddings = query_embeddings + last_hidden_states.sum(1)

        return EasyDict(
            {
                "pooler_output": query_embeddings,
            }
        )
