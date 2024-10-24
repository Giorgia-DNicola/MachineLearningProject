from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import accuracy_score, classification_report


def evaluate_gpt(test_data):
    model = GPT2LMHeadModel.from_pretrained('./models/fine_tuned_gpt/')
    tokenizer = GPT2Tokenizer.from_pretrained('./models/tokenizer/')

    correct_labels = []
    predictions = []

    for text, label in test_data:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=-1)

        correct_labels.append(label)
        predictions.append(pred_label.item())

    print(classification_report(correct_labels, predictions))
