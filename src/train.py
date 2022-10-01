# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from dataset import CustomDataset
from models import BertCLSModel
from train_loop import train
from utils import set_seed, load_params, make_history_table, plot_history

seed = 0

pretrained_model_name = 'izumi-lab/electra-base-japanese-discriminator'
max_length = 512
trunc = 'last'
batch_size = 16

n_classes = 2
dropout_rate = 0.1
is_freeze = False

learning_rate = 2e-5
n_epochs = 4


def main():
    set_seed(seed=seed)

    params = load_params('settings.json')
    print(params)
    input_dir = params['RAW_DATA_DIR']
    output_dir_log = params['LOGS_DIR']
    output_dir_model = params['MODEL_CHECKPOINT_DIR']

    os.makedirs(output_dir_log, exist_ok=True)
    os.makedirs(output_dir_model, exist_ok=True)

    df_train = pd.read_csv(os.path.join(input_dir, 'train.csv'))

    class_weight_train = compute_class_weight('balanced',
                                              classes=np.unique(df_train.isFake.values),
                                              y=df_train.isFake.values)
    print(class_weight_train)

    # 訓練データ
    X_train_ = df_train.text.ravel()
    y_train_ = df_train.isFake.ravel()

    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_train_, y_train_, test_size=0.2, random_state=seed)

    # Model
    model = BertCLSModel(pretrained_model_name=pretrained_model_name,
                         n_classes=n_classes,
                         dropout_rate=dropout_rate,
                         is_freeze=is_freeze)

    # Dataset
    dataset_train = CustomDataset(X=X_train,
                                  y=y_train,
                                  pretrained_model_name=pretrained_model_name,
                                  max_length=max_length, trunc=trunc)
    dataset_valid = CustomDataset(X=X_valid,
                                  y=y_valid,
                                  pretrained_model_name=pretrained_model_name,
                                  max_length=max_length, trunc=trunc)

    # DataLoader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size,
                                  shuffle=False)

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    # Loss function
    weight = torch.tensor(class_weight_train, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = None

    # Training
    history = train(model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    n_epochs=n_epochs,
                    dataloader_train=dataloader_train,
                    dataloader_valid=dataloader_valid)

    # Save model
    model_file_name = os.path.join(output_dir_model, 'model.pt')
    torch.save(model.state_dict(), model_file_name)

    # Save history
    df_history_train = make_history_table(history, type='train')
    df_history_valid = make_history_table(history, type='valid')
    df_history_train.to_csv(os.path.join(output_dir_log, 'history_train.csv'))
    df_history_valid.to_csv(os.path.join(output_dir_log, 'history_valid.csv'))
    plot_history(n_epochs, history, output_dir_log)


if __name__ == '__main__':
    main()
