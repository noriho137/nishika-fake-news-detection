# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self, X, y=None, pretrained_model_name=None, max_length=512, trunc='last'):
        self.X = X
        self.y = y
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name)
        self.max_length = max_length
        self.trunc = trunc

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = self.X[index]

        if self.trunc == 'first':
            ret = self.truncate_first(text=text)
        elif self.trunc == 'middle':
            ret = self.truncate_middle(text=text)
        elif self.trunc == 'last':
            ret = self.truncate_last(text=text)
        else:
            raise Exception(f'Invalid argument: trunc={self.trunc}')

        if self.y is not None:
            # label = torch.LongTensor([self.y[index]])
            # ret = (ret, label)
            ret['label'] = torch.LongTensor([self.y[index]])
        return ret

    def truncate_first(self, text):
        inputs = self.tokenizer(text=text,
                                add_special_tokens=False,
                                return_token_type_ids=False,
                                return_attention_mask=False,
                                padding=False,
                                truncation=False)
        input_ids = inputs.get('input_ids')

        input_ids = input_ids[-(self.max_length - 2):]
        input_ids.insert(0, self.tokenizer.cls_token_id)
        input_ids.append(self.tokenizer.sep_token_id)
        token_length = len(input_ids)
        input_ids = input_ids + [0] * (self.max_length - token_length)
        attention_mask = [1] * token_length + [0] * (self.max_length - token_length)
        token_type_ids = [0] * self.max_length

        ret = {'input_ids': torch.LongTensor(input_ids),
               'token_type_ids': torch.LongTensor(token_type_ids),
               'attention_mask': torch.LongTensor(attention_mask)}
        return ret

    def truncate_middle(self, text):
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                return_token_type_ids=True,
                                return_attention_mask=True,
                                padding=False,
                                truncation=False)
        n_tokens = len(inputs.get('input_ids'))

        if n_tokens == self.max_length:
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
            token_type_ids = inputs.get('token_type_ids')
        elif n_tokens < self.max_length:
            pad = [0 for _ in range(self.max_length - n_tokens)]
            input_ids = inputs.get('input_ids') + pad
            attention_mask = [1 if i < n_tokens else 0 for i in range(self.max_length)]
            token_type_ids = [0 if i < n_tokens else 0 for i in range(self.max_length)]
        else:
            half_length = self.max_length // 2
            input_ids_ = inputs.get('input_ids')
            input_ids = input_ids_[:half_length] + input_ids_[-half_length:]
            attention_mask = [1 for _ in range(self.max_length)]
            token_type_ids = [0 for _ in range(self.max_length)]

        ret = {'input_ids': torch.LongTensor(input_ids),
               'token_type_ids': torch.LongTensor(token_type_ids),
               'attention_mask': torch.LongTensor(attention_mask)}
        return ret

    def truncate_last(self, text):
        inputs = self.tokenizer(text=text,
                                add_special_tokens=True,
                                return_token_type_ids=True,
                                return_attention_mask=True,
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True)
        input_ids = inputs.get('input_ids')
        token_type_ids = inputs.get('token_type_ids')
        attention_mask = inputs.get('attention_mask')

        ret = {'input_ids': torch.LongTensor(input_ids),
               'token_type_ids': torch.LongTensor(token_type_ids),
               'attention_mask': torch.LongTensor(attention_mask)}
        return ret
