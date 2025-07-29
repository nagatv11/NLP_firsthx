import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from preprocess import preprocess_text
from utils import encode_labels, compute_metrics, ChiefComplaintDataset

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

def train():
    df = pd.read_csv("data/synthetic_data.csv")  # assumes columns: statements, chief_complaint
    df["statements"] = df["statements"].apply(preprocess_text)
    all_labels = sorted(set(label for sublist in df["chief_complaint"].str.split(";") for label in sublist))
    label_map = {label: i for i, label in enumerate(all_labels)}
    y = encode_labels(df["chief_complaint"], label_map)

    X_train, X_val, y_train, y_val = train_test_split(df["statements"], y, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ChiefComplaintDataset(X_train.tolist(), y_train, tokenizer)
    val_ds = ChiefComplaintDataset(X_val.tolist(), y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map),
                                                                problem_type="multi_label_classification")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            outputs.loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                all_preds.extend(probs)
                all_true.extend(batch['labels'].cpu().numpy())
        metrics = compute_metrics(np.array(all_preds), np.array(all_true), threshold=0.5)
        print(f"F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    model.save_pretrained("models/classifier")
    tokenizer.save_pretrained("models/classifier")

if __name__ == "__main__":
    train()
