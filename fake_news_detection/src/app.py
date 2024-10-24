from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained('./models/fine_tuned_gpt/')
tokenizer = GPT2Tokenizer.from_pretrained('./models/tokenizer/')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    tokens = tokenizer(data, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**tokens)
    pred = torch.argmax(outputs.logits, dim=-1)
    is_fake = bool(pred.item())
    return jsonify({'fake_news': is_fake})

if __name__ == '__main__':
    app.run(debug=True)
