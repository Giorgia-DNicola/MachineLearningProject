from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader
import torch


def train_gpt(train_data, epochs=3, learning_rate=5e-5):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs = tokenizer(batch['text'], return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained('./models/fine_tuned_gpt/')
