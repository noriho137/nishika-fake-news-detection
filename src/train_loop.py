# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


def train(model, loss_fn, optimizer, scheduler, n_epochs,
          dataloader_train, dataloader_valid):
    loader_dict = {'train': dataloader_train, 'valid': dataloader_valid}

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Device: {device}')

    model.to(device)

    history = {'train_loss': [], 'train_accuracy': [],
               'train_precision': [], 'train_recall': [], 'train_f1': [],
               'train_cm': [],
               'valid_loss': [], 'valid_accuracy': [],
               'valid_precision': [], 'valid_recall': [], 'valid_f1': [],
               'valid_cm': []}

    for epoch in range(n_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            t_epoch_start = time.time()

            running_loss = 0.0

            y_pred = np.array([])
            y_true = np.array([])

            for batch in loader_dict[phase]:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
                    _, predicted_labels = torch.max(outputs, dim=1)

                    loss = loss_fn(outputs, labels.squeeze(1))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                running_loss += loss.item()

                y_true = np.append(y_true, labels.squeeze(1).to('cpu').detach().numpy().copy())
                y_pred = np.append(y_pred, predicted_labels.to('cpu').detach().numpy().copy())

            t_epoch_end = time.time()
            elapsed = t_epoch_end - t_epoch_start

            epoch_loss = running_loss / len(loader_dict[phase])

            epoch_accuracy = accuracy_score(y_true, y_pred)
            epoch_precision = precision_score(y_true, y_pred)
            epoch_recall = recall_score(y_true, y_pred)
            epoch_f1 = f1_score(y_true, y_pred)
            epoch_cm = confusion_matrix(y_true, y_pred)

            print(f'Epoch {epoch + 1}/{n_epochs} [{phase}] '
                  f'loss={epoch_loss}, cm={epoch_cm.ravel()}, '
                  f'accuracy={epoch_accuracy}, precision={epoch_precision}, '
                  f'recall={epoch_recall}, f1={epoch_f1}, '
                  f'time={elapsed}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_accuracy'].append(epoch_accuracy)
                history['train_precision'].append(epoch_precision)
                history['train_recall'].append(epoch_recall)
                history['train_f1'].append(epoch_f1)
                history['train_cm'].append(epoch_cm)
            else:
                history['valid_loss'].append(epoch_loss)
                history['valid_accuracy'].append(epoch_accuracy)
                history['valid_precision'].append(epoch_precision)
                history['valid_recall'].append(epoch_recall)
                history['valid_f1'].append(epoch_f1)
                history['valid_cm'].append(epoch_cm)

    return history
