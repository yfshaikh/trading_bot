from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load the tokenizer and model for FinBERT (pre-trained model for financial sentiment analysis)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# fefine the possible sentiment labels
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        # tokenize the input news headlines, converting them into tensors (for the model to process)
        # 'padding=True' ensures the tokenized inputs are of equal length
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        # pass the tokens through the model to get the logits (raw model outputs)
        # 'input_ids' are the tokenized representations of the text
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]

        # sum the logits across the news headlines, then apply softmax to get probabilities
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)

        # get the highest probability and corresponding sentiment label
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        # if no news headlines are provided, return 0 probability and "neutral" sentiment
        return 0, labels[-1]


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!','traders were displeased!'])
    print(tensor, sentiment)
    print(torch.cuda.is_available())