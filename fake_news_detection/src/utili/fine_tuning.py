import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import pickle
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

# Carichiamo i dati tokenizzati salvati
tokenized_data_path = Path("../../data/tokenized/tokenized_data.pkl")
if not tokenized_data_path.exists():
    raise FileNotFoundError("File dei dati tokenizzati non trovato.")

with open(tokenized_data_path, "rb") as f:
    data = pickle.load(f)

train_encodings, train_labels = data["train_encodings"], data["train_labels"]
test_encodings, test_labels = data["test_encodings"], data["test_labels"]

# Convertiamo i dati in formato Dataset per Hugging Face
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

# Carichiamo il modello RoBERTa pre-addestrato per la classificazione binaria
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Definiamo la funzione di valutazione
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Configuriamo i parametri di training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1, #con valore 3 l'addestramento dura troppo (circa 3 giorni)
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,  # Early stopping
    metric_for_best_model="f1"
)

# Creiamo il Trainer e avviamo il training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Salviamo il modello addestrato
model_save_path = Path("../../models/fine_tuned_roberta")
model_save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(model_save_path)
print(f"Modello fine-tuned salvato in: {model_save_path}")