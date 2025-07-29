import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def encode_labels(labels, label_map):
    encoded = []
    for label_list in labels.str.split(";"):
        binary = [0] * len(label_map)
        for label in label_list:
            if label in label_map:
                binary[label_map[label]] = 1
        encoded.append(binary)
    return encoded

class ChiefComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], padding='max_length', truncation=True,
                                max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.texts)

def compute_metrics(preds, labels, threshold=0.5):
    preds_bin = (preds >= threshold).astype(int)
    return {
        "f1": f1_score(labels, preds_bin, average="micro"),
        "precision": precision_score(labels, preds_bin, average="micro"),
        "recall": recall_score(labels, preds_bin, average="micro")
    }
