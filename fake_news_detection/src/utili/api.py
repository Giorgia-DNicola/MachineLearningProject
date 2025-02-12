from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

# Inizializza FastAPI
app = FastAPI(
    title="Fake News Detection API",
    description="API per la classificazione di notizie come vere o false.",
    version="1.0"
)

# Percorso del modello fine-tuned
MODEL_PATH = Path("../../models/fine_tuned_roberta")
if not MODEL_PATH.exists():
    raise FileNotFoundError("Modello fine-tuned non trovato.")

# Carica il modello e il tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Definizione dello schema per le richieste
class NewsArticle(BaseModel):
    text: str

# Endpoint per la predizione
@app.post("/predict")
def predict(article: NewsArticle):
    try:
        # Tokenizzazione del testo
        inputs = tokenizer(article.text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Eseguiamo la predizione
        with torch.no_grad():
            outputs = model(**inputs)

        # Interpretazione del risultato
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

        # 1 = Vero, 0 = Falso
        label = "Real News" if prediction == 1 else "Fake News"

        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la predizione: {e}")