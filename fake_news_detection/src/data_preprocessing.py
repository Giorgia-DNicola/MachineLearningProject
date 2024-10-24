from transformers import GPT2Tokenizer
import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Preprocessing steps like removing punctuation, lowering case, etc.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    df['tokens'] = df['text'].apply(lambda x: tokenizer(x, padding=True, truncation=True))
    df.to_csv('data/processed/processed_news.csv', index=False)
