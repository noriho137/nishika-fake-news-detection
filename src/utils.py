# -*- coding: utf-8 -*-
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed=0):
    # for Python
    random.seed(seed)

    # for NumPy
    np.random.seed(seed)

    # for PyTorch, CUDA and cuDNN
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_params(params_file):
    with open(params_file, mode='r') as f:
        params = json.load(f)
    return params


def make_history_table(history, type='train'):
    loss = history[f'{type}_loss']
    cm = history[f'{type}_cm']
    acc = history[f'{type}_accuracy']
    pr = history[f'{type}_precision']
    rc = history[f'{type}_recall']
    f1 = history[f'{type}_f1']

    losses, tns, fps, fns, tps, accs, prs, rcs, f1s = [], [], [], [], [], [], [], [], []

    for loss, cm, acc, pr, rc, f1 in zip(loss, cm, acc, pr, rc, f1):
        losses.append(loss)
        tn, fp, fn, tp = cm.ravel()
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        accs.append(acc)
        prs.append(pr)
        rcs.append(rc)
        f1s.append(f1)

    df_history = pd.DataFrame({'Loss': losses,
                               'TN': tns,
                               'FP': fps,
                               'FN': fns,
                               'TP': tps,
                               'Accuracy': accs,
                               'Precision': prs,
                               'Recall': rcs,
                               'F1': f1s},
                              index=[f'epoch {i+1}' for i in range(len(history['train_cm']))])
    return df_history


def plot_history(n_epochs, history, output_dir):
    targets = ['loss', 'accuracy', 'precision', 'recall', 'f1']

    fig = plt.figure(figsize=(30, 5))

    for i, target in enumerate(targets, 1):
        ax = fig.add_subplot(1, 5, i)
        ax.plot(range(1, n_epochs + 1), history[f'train_{target}'], label='train')
        ax.plot(range(1, n_epochs + 1), history[f'valid_{target}'], label='valid')
        ax.set_xlabel('epoch')
        ax.set_ylabel(f'{target}')
        ax.set_title(f'{target}')
        ax.grid()
        ax.legend()

    plt.show()
    fig.savefig(os.path.join(output_dir, 'history.png'))
