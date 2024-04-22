import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Config
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from easydict import EasyDict

class RetrieverT5(torch.nn.Module):
    """
    Class of retriever model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]

        QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]
        query_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion)
        self.query_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion, config=query_model_config)
        
        self.SEP_ENCODER = True if 'separate_query_and_item_encoders' in self.config.model_config.modules else None
        self.SPAN_DETECTOR = True if 'span_detector' in self.config.model_config.modules else None

        if self.SEP_ENCODER:
            ItemEncoderModelClass = globals()[self.config.model_config.ItemEncoderModelClass]

            ItemEncoderConfigClass = globals()[self.config.model_config.ItemEncoderConfigClass]
            item_model_config = ItemEncoderConfigClass.from_pretrained(self.config.model_config.ItemEncoderModelVersion)
            self.item_encoder = ItemEncoderModelClass.from_pretrained(self.config.model_config.ItemEncoderModelVersion, config=item_model_config)
        else:
            # Use the same model for query and item encoders
            item_model_config = query_model_config
            self.item_encoder = self.query_encoder
        

        if self.SPAN_DETECTOR:
            # add span detection layer
            self.span_outputs = nn.Linear(item_model_config.hidden_size, 2)

        if self.config.model_config.get('pooling_output', None) is not None:
            self.query_pooler = nn.Sequential(
                nn.Linear(query_model_config.hidden_size, self.config.model_config.pooling_output.dim),
                nn.Dropout(self.config.model_config.pooling_output.dropout)
            )
            self.item_pooler = nn.Sequential(
                nn.Linear(item_model_config.hidden_size, self.config.model_config.pooling_output.dim),
                nn.Dropout(self.config.model_config.pooling_output.dropout)
            )
        else:
            self.query_pooler = None
            self.item_pooler = None
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        

    
    def resize_token_embeddings(self, dim):
        self.query_encoder.resize_token_embeddings(dim)
        if 'separate_query_and_item_encoders' in self.config.model_config.modules:
            self.item_encoder.resize_token_embeddings(dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        item_input_ids=None,
        item_attention_mask=None,
        labels=None,
        span_labels=None,
        **kwargs
    ):
        # query encoder
        query_outputs = self.query_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        query_last_hidden_states = query_outputs.last_hidden_state
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states[:, 0]
        # print('query_embeddings', query_embeddings.shape)

        # item encoder
        item_outputs = self.item_encoder(input_ids=item_input_ids,
                                            attention_mask=item_attention_mask)
        item_last_hidden_states = item_outputs.last_hidden_state
        if self.item_pooler is not None:
            item_last_hidden_states = self.item_pooler(item_last_hidden_states)
        item_embeddings = item_last_hidden_states[:, 0]
        # print('item_embeddings', item_embeddings.shape)
        

        


        ################## in-batch negative sampling ###############
        batch_size = query_embeddings.shape[0]
        batch_size_with_pos_and_neg = item_embeddings.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos
        
        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size  
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(labels.device)
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step*i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)

        in_batch_scores = torch.matmul(query_embeddings, item_embeddings.T)
        loss = self.loss_fn(in_batch_scores, in_batch_labels)
        # print('loss', loss)
        # input('forwarded.')


        # sequence_output = item_last_hidden_states

        # logits = self.span_outputs(sequence_output)
        # print('logits', logits)
        # start_logits, end_logits = logits.split(1, dim=-1)
        # print(start_logits.shape, end_logits.shape)
        # start_logits = start_logits.squeeze(-1).contiguous()
        # end_logits = end_logits.squeeze(-1).contiguous()
        # print(start_logits.shape, end_logits.shape)
        # input()
        # total_loss = None
        # if span_labels is not None:
        #     start_positions = span_labels[:, 0]
        #     end_positions = span_labels[:, 1]

        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_positions.size()) > 1:
        #         start_positions = start_positions.squeeze(-1)
        #     if len(end_positions.size()) > 1:
        #         end_positions = end_positions.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = start_logits.size(1)
        #     start_positions = start_positions.clamp(0, ignored_index)
        #     end_positions = end_positions.clamp(0, ignored_index)

        #     loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        #     start_loss = loss_fct(start_logits, start_positions)
        #     end_loss = loss_fct(end_logits, end_positions)
        #     total_loss = (start_loss + end_loss) / 2


        return EasyDict({
            'loss': loss,
        })

    def generate_query_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        # query encoder
        query_outputs = self.query_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        query_last_hidden_states = query_outputs.last_hidden_state
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states[:, 0]
        return query_embeddings

    def generate_item_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        # item encoder
        item_outputs = self.item_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        item_last_hidden_states = item_outputs.last_hidden_state
        if self.item_pooler is not None:
            item_last_hidden_states = self.item_pooler(item_last_hidden_states)
        item_embeddings = item_last_hidden_states[:, 0]
        return item_embeddings
    
    
    def create_bpr_loss(self, query, pos_items, neg_items):
        """[summary]

        Args:
            query ([type]): batch_size x hidden_size
            pos_items ([type]): batch_size x hidden_size
            neg_items ([type]): batch_size*num_neg_samples x hidden_size

        Returns:
            [type]: [description]
        """
        batch_size = query.shape[0]
        hidden_size = query.shape[1]
        num_neg_samples = neg_items.shape[0] // batch_size

        # extend the query for mapping with any number of neg samples
        extend_query = query.repeat(1, num_neg_samples).reshape(-1, hidden_size)

        pos_scores = torch.sum(torch.mul(query, pos_items), axis=1) # batch_size
        if num_neg_samples > 1:
            # extend pos_scores to match with neg scores
            pos_scores = pos_scores.repeat(num_neg_samples, 1).permute(1,0).reshape(-1)
        # print('pos_scores', pos_scores)
        neg_scores = torch.sum(torch.mul(extend_query, neg_items), axis=1)
        # print('neg_scores', neg_scores)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        
        return mf_loss
