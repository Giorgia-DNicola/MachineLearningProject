import torch
from transformers import AutoModelForSequenceClassification, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import pickle
from datasets import Dataset

# Percorso del modello addestrato
model_path = Path("../../models/fine_tuned_roberta")

# Carichiamo il modello salvato
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Carichiamo i dati di test
tokenized_data_path = Path("../../data/tokenized/tokenized_data.pkl")
with open(tokenized_data_path, "rb") as f:
    data = pickle.load(f)

test_encodings, test_labels = data["test_encodings"], data["test_labels"]

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

# Definizione della funzione di valutazione
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Creiamo un Trainer per la valutazione
training_args = {"output_dir": "./results"}
trainer = Trainer(model=model, compute_metrics=compute_metrics)

# Eseguiamo la valutazione
results = trainer.evaluate(test_dataset)
print("Risultati del test set:", results)
