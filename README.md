# FirstHx Symptom Classifier (BioClinicalBERT + sBERT)
A multi-label classification and similarity-based NLP pipeline for identifying chief complaints from patient free-text.

## Models
- 🔬 Classification: BioClinicalBERT
- 🔍 Similarity: sBERT - multi-qa-MiniLM-L6-cos-v1

## Getting Started
```bash
git clone https://github.com/nagatv11/firsthx-ml.git
cd firsthx-ml
pip install -r requirements.txt
```

## Train the Classifier
```bash
python train.py
```

## Run Prediction
```bash
python predict.py --text "I have chest pain and shortness of breath"
```

