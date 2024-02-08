from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


class EntityMatcher:
    def __init__(self, model_path, tokenizer_path='distilbert-base-uncased'):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, entity_1, entity_2):
        inputs = self.tokenizer(entity_1, entity_2, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            return prediction, probabilities[0][prediction].item()
