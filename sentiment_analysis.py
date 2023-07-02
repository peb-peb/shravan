from transformers import pipeline

def sentiment_analyser(text):
  sent = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
  return sent(text)
