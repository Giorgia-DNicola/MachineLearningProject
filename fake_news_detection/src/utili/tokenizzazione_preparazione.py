from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pickle
from pathlib import Path
import pandas as pd

# Carica il dataset pre-processato
processed_file_path = Path("../../data/processed/processed_news.csv")
if not processed_file_path.exists():
    raise FileNotFoundError("File del dataset pre-processato non trovato.")

df_combined = pd.read_csv(processed_file_path)

# Suddivisione del dataset in training (80%) e testing (20%)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_combined["text"], df_combined["label"], test_size=0.2, random_state=42
)

# Carichiamo il tokenizer pre-addestrato di RoBERTa
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Tokenizziamo i testi (troncamento e padding per uniformare la lunghezza)
try:
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)
except Exception as e:
    raise ValueError(f"Errore durante la tokenizzazione: {e}")

# Salviamo i dati tokenizzati in formato pickle per un uso futuro
tokenized_data_path = Path("../../data/tokenized/tokenized_data.pkl")
tokenized_data_path.parent.mkdir(parents=True, exist_ok=True)  # Crea la cartella se non esiste

with open(tokenized_data_path, "wb") as f:
    pickle.dump({
        "train_encodings": train_encodings,
        "train_labels": train_labels,
        "test_encodings": test_encodings,
        "test_labels": test_labels
    }, f)

print(f"Dati tokenizzati salvati in: {tokenized_data_path}")