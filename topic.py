from transformers import pipeline

def topic_gen(text):
  topic = pipeline("text2text-generation", model="knkarthick/TOPIC-DIALOGSUM")
  return topic(text)
