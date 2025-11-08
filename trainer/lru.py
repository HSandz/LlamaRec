from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import pickle
import numpy as np
from abc import *
from pathlib import Path


class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
    
    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, exclude_history=True):
        seqs, labels = batch
        
        scores = self.model(seqs)[:, -1, :]
        B, L = seqs.shape
        if exclude_history:
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics
    
    def generate_candidates(self, retrieved_data_path):
        self.model.eval()
        val_probs, val_labels = [], []
        test_probs, test_labels = [], []
        with torch.no_grad():
            print('*************** Generating Candidates for Validation Set ***************')
            tqdm_dataloader = tqdm(self.val_loader)
            val_dataset = self.val_loader.dataset
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                
                # Mask ALL history items for each user, not just items in the truncated sequence
                for i in range(B):
                    user_idx = batch_idx * self.val_loader.batch_size + i
                    if user_idx < len(val_dataset.users):
                        user_id = val_dataset.users[user_idx]
                        # Get ALL training items for this user (not just last max_len)
                        all_history_items = val_dataset.u2seq[user_id]
                        # Mask all history items
                        scores[i, all_history_items] = -1e9
                
                scores[:, 0] = -1e9  # padding
                val_probs.extend(scores.tolist())
                val_labels.extend(labels.view(-1).tolist())
            val_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(val_probs), 
                                                          torch.tensor(val_labels).view(-1), self.metric_ks)
            print(val_metrics)

            print('****************** Generating Candidates for Test Set ******************')
            tqdm_dataloader = tqdm(self.test_loader)
            test_dataset = self.test_loader.dataset
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                
                # Mask ALL history items for each user (train + val)
                for i in range(B):
                    user_idx = batch_idx * self.test_loader.batch_size + i
                    if user_idx < len(test_dataset.users):
                        user_id = test_dataset.users[user_idx]
                        # Get ALL training + validation items for this user
                        all_history_items = test_dataset.u2seq[user_id] + test_dataset.u2val[user_id]
                        # Mask all history items
                        scores[i, all_history_items] = -1e9
                
                scores[:, 0] = -1e9  # padding
                test_probs.extend(scores.tolist())
                test_labels.extend(labels.view(-1).tolist())
            test_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(test_probs), 
                                                           torch.tensor(test_labels).view(-1), self.metric_ks)
            print(test_metrics)

        with open(retrieved_data_path, 'wb') as f:
            pickle.dump({'val_probs': val_probs,
                         'val_labels': val_labels,
                         'val_metrics': val_metrics,
                         'test_probs': test_probs,
                         'test_labels': test_labels,
                         'test_metrics': test_metrics}, f)