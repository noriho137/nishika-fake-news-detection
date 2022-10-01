# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CustomDataset
from models import BertCLSModel
from utils import set_seed, load_params


seed = 0

pretrained_model_name = 'izumi-lab/electra-base-japanese-discriminator'
max_length = 512
trunc = 'last'
batch_size = 32

n_classes = 2
dropout_rate = 0.1
is_freeze = False


def main():
    set_seed(seed=seed)

    params = load_params('settings.json')
    input_dir_data = params['RAW_DATA_DIR']
    input_dir_model = params['MODEL_CHECKPOINT_DIR']
    output_dir_submission = params['SUBMISSION_DIR']

    os.makedirs(output_dir_submission, exist_ok=True)

    df_test = pd.read_csv(os.path.join(input_dir_data, 'test.csv'))
    X_test = df_test.text.ravel()

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device={device}')

    # Load trained Model
    model_file = os.path.join(input_dir_model, 'model.pt')
    model = BertCLSModel(pretrained_model_name=pretrained_model_name,
                         n_classes=n_classes,
                         dropout_rate=dropout_rate,
                         is_freeze=is_freeze)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    # Dataset
    dataset_test = CustomDataset(X=X_test, y=None,
                                 pretrained_model_name=pretrained_model_name,
                                 max_length=max_length, trunc=trunc)

    # DataLoader
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=batch_size,
                                 shuffle=False)

    print('evaluation start')

    model.to(device)

    model.eval()

    y_predict_all = np.array([])

    for X in dataloader_test:
        input_ids = X['input_ids'].to(device)
        token_type_ids = X['token_type_ids'].to(device)
        attention_mask = X['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids, attention_mask)
            y_predict = torch.argmax(outputs, dim=-1).cpu().numpy()
            y_predict_all = np.append(y_predict_all, y_predict)

    print('evaluation end')

    # Make submission file
    df_submit = pd.DataFrame({'id': df_test.id.values, 'isFake': y_predict_all})
    df_submit['isFake'] = df_submit['isFake'].astype(int)

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    submit_file_name = f'submit_{timestamp}.csv'
    print(submit_file_name)

    df_submit.to_csv(os.path.join(output_dir_submission, submit_file_name), index=False, header=True)


if __name__ == '__main__':
    main()
