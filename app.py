import gradio as gr
import requests
# from transcribe import transcribe
from sentiment_analysis import sentiment_analyser
from summary import summarizer
from topic import topic_gen
from data import data

def transcribe2():
  response = requests.post("https://dwarkesh-whisper-speaker-recognition.hf.space/run/predict", json={
    "data": [
      {"name":"audio.wav","data":"data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="},
      2,
  ]}).json()

  data = response["data"]

def main(audio_file, number_of_speakers):
  # Audio to Text Converter
  # text_data = transcribe(audio_file, number_of_speakers)
  # print(text_data)
  text_data = data
  topic = topic_gen(text_data)[0]["generated_text"]
  summary = summarizer(text_data)[0]["summary_text"]
  sent_analy = sentiment_analyser(text_data)
  sent_analysis = sent_analy[0]["label"] + " (" + str(float(sent_analy[0]["score"]) * 100) + "%)"
  return topic, summary, sent_analysis

# UI Interface on the Hugging Face Page
with gr.Blocks() as demo:
  gr.Markdown("# Shravan - Unlocking Value from Call Data")
  with gr.Box():
    with gr.Row():
      with gr.Column():
        audio_file = gr.Audio(label="Upload an Audio file (.wav)", source="upload", type="filepath")
        number_of_speakers = gr.Number(label="Number of Speakers", value=2)
        with gr.Row():
          btn_clear = gr.ClearButton(value="Clear", components=[audio_file, number_of_speakers])
          btn_submit = gr.Button(value="Submit")
      with gr.Column():
        topic = gr.Textbox(label="Title", placeholder="Title for Conversation")
        summary = gr.Textbox(label="Short Summary", placeholder="Short Summary for Conversation")
        sentiment_analysis = gr.Textbox(label="Sentiment Analysis", placeholder="Sentiment Analysis for Conversation")
      btn_submit.click(fn=main, inputs=[audio_file, number_of_speakers], outputs=[topic, summary, sentiment_analysis])
    gr.Markdown("## Examples")
    gr.Examples( 
      examples=[
        ["./examples/sample4.wav", 2],
      ],
      inputs=[audio_file, number_of_speakers],
      outputs=[topic, summary, sentiment_analysis],
      fn=main,
    )
  gr.Markdown(
    """
    NOTE: The Tool takes around 5mins to run. So be patient! ;)
    See [https://github.com/peb-peb/shravan](https://github.com/peb-peb/shravan) for more details.
    """
  )

demo.launch()
