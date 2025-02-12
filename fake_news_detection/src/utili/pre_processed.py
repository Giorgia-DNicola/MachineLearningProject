import pandas as pd
from pathlib import Path

# Percorsi dei file caricati
true_path = Path("../../data/raw/True/True.csv")
fake_path = Path("../../data/raw/Fake/Fake.csv")

# Verifica che i file esistano
if not true_path.exists() or not fake_path.exists():
    raise FileNotFoundError("Uno o più file non trovati.")

# Caricamento dei dataset
df_true = pd.read_csv(true_path)
df_fake = pd.read_csv(fake_path)

# Aggiungiamo le etichette ai dataset
df_true["label"] = 1  # Notizie vere -> 1
df_fake["label"] = 0  # Notizie false -> 0

# Uniamo i due dataset in un unico dataframe
df_combined = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)

# Rimuoviamo colonne non necessarie
df_combined = df_combined.drop(columns=["subject", "date"])

# Rimuoviamo eventuali duplicati
df_combined = df_combined.drop_duplicates()

# Pulizia del testo (opzionale)
df_combined["text"] = df_combined["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

# Salviamo il dataset pre-processato in un nuovo file CSV
processed_file_path = Path("../../data/processed/processed_news.csv")
processed_file_path.parent.mkdir(parents=True, exist_ok=True)  # Crea la cartella se non esiste
df_combined.to_csv(processed_file_path, index=False)

print(f"Dataset pre-processato salvato in: {processed_file_path}")