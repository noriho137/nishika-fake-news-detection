# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel


class BertCLSModel(nn.Module):
    def __init__(self, pretrained_model_name, n_classes, dropout_rate=0.0,
                 is_freeze=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name,
            return_dict=True)
        self.bert.config.output_hidden_states = True
        self.is_freeze = is_freeze
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 4, n_classes)

        if self.is_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        out = torch.cat([outputs.get('hidden_states')[-1 * i][:, 0] for i in range(1, 4 + 1)], dim=1)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits
