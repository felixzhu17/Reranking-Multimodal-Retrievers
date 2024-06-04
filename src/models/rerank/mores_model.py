"""
MORES on BERT base.
"""

# https://github.com/luyug/MORES/blob/dev/bert_mores.py
import os
import warnings

import torch
import torch.functional as F
import copy
from transformers import BertModel, BertConfig, AutoModel
from transformers.models.bert.modeling_bert import BertPooler, BertPreTrainingHeads
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from torch import nn

from bert_mores import MORES_BertLayer
from arguments import ModelArguments, DataArguments, \
    MORESTrainingArguments as TrainingArguments
from torch import nn
from transformers.models.bert.modeling_bert import  apply_chunking_to_forward, BertLayer


class MORES_BertLayer(BertLayer):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        self_attention_outputs = self.attention(
            attention_output,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = outputs + self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs


class MORESSym(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.is_decoder = True
        config.add_cross_attention = True
        self.interaction_module = nn.ModuleList(
            [MORES_BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.proj = nn.Linear(config.hidden_size, 2)

    def forward(self, qry, doc, qry_mask, cross_mask, attention_adj = None):

        qry_mask = self.get_extended_attention_mask(qry_mask, qry_mask.size(), qry.device)
        cross_mask = self.get_extended_attention_mask(cross_mask, doc.size(), qry.device)

        hidden_states = qry
        for i, ib_layer in enumerate(self.interaction_module):
            layer_outputs = ib_layer(
                hidden_states,
                attention_mask=qry_mask,
                encoder_hidden_states=doc,
                encoder_attention_mask=cross_mask,
            )
            hidden_states = layer_outputs[0]

        cls_reps = hidden_states[:, 0]

        # Pass the CLS token's output through the classifier to get the logits
        logits1 = self.classifier1(cls_reps)
        logits2 = self.classifier2(cls_reps)

        return logits1, logits2
    
    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Make an extended attention mask for attention masking.
        """
        # Prepare the attention mask
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for attention_mask (shape {})".format(attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask