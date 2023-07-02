from transformers import pipeline

def summarizer(text):
  summ = pipeline("summarization", model="knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM")
  return summ(text)
