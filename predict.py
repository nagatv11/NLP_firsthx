import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import preprocess_text
import numpy as np
import argparse

def predict(text, top_k=5, threshold=0.5):
    tokenizer = AutoTokenizer.from_pretrained("models/classifier")
    model = AutoModelForSequenceClassification.from_pretrained("models/classifier")
    model.eval()

    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    label_map = [f"label_{i}" for i in range(len(probs))]
    output = {label: prob for label, prob in zip(label_map, probs) if prob >= threshold}
    top_output = dict(sorted(output.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return top_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    print(predict(args.text))
