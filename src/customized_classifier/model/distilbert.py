import math

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers.file_utils import ModelOutput

@dataclass
class SequenceClassifierDoubleOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_v2: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CustomizedDistilbertClassifier(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_labels_2 = config.num_labels_2
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier_v1 = nn.Linear(config.dim, config.num_labels)
        self.classifier_v2 = nn.Linear(config.dim, config.num_labels_2)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier_v1(pooled_output)  # (bs, num_labels)  #label 1 classifier
        logits_2 = self.classifier_v2(pooled_output)  # (bs, num_labels_2)  #label 2 classifier

        loss = None
        if labels is not None:
            self.config.problem_type = "single_label_classification"
            loss_fct = CrossEntropyLoss()
            loss_v1 = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if labels_2 is not None:
            loss_fct_v2 = CrossEntropyLoss()
            loss_v2 = loss_fct_v2(logits_2.view(-1, self.num_labels_2), labels_2.view(-1))

        if labels is not None and labels_2 is not None:
            loss = (loss_v1 + loss_v2) / 2  #simple average to get the lost, this can also be customized
        elif labels is not None:
            loss = loss_v1
        elif labels_2 is not None:
            loss = loss_v2

        if not return_dict:
            output = (logits,logits_2,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierDoubleOutput(
            loss=loss,
            logits=logits,
            logits_2=logits_2,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    