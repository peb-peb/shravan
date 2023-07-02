import whisper
import datetime
import subprocess
import wave
import contextlib


import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import numpy as np

model = whisper.load_model("large-v2")
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

def transcribe(audio, num_speakers):
  print(type(audio))
  path, error = convert_to_wav(audio)
  if error is not None:
    return error

  duration = get_duration(path)
  if duration > 4 * 60 * 60:
    return "Audio duration too long"

  result = model.transcribe(path)
  segments = result["segments"]

  num_speakers = min(max(round(num_speakers), 1), len(segments))
  if len(segments) == 1:
    segments[0]['speaker'] = 'SPEAKER 1'
  else:
    embeddings = make_embeddings(path, segments, duration)
    add_speaker_labels(segments, embeddings, num_speakers)
  output = get_output(segments)
  return output

def convert_to_wav(path):
  if path[-3:] != 'wav':
    new_path = '.'.join(path.split('.')[:-1]) + '.wav'
    try:
      subprocess.call(['ffmpeg', '-i', path, new_path, '-y'])
    except:
      return path, 'Error: Could not convert file to .wav'
    path = new_path
  return path, None

def get_duration(path):
  with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    return frames / float(rate)

def make_embeddings(path, segments, duration):
  embeddings = np.zeros(shape=(len(segments), 192))
  for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(path, segment, duration)
  return np.nan_to_num(embeddings)

audio = Audio()

def segment_embedding(path, segment, duration):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

def add_speaker_labels(segments, embeddings, num_speakers):
  """Add speaker labels"""
  clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
  labels = clustering.labels_
  for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

def time(secs):
  """Function to return time delta"""
  return datetime.timedelta(seconds=round(secs))

def get_output(segments):
  """Format and generate the output string"""
  output = ''
  for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
      if i != 0:
        output += '\n\n'
      output += segment["speaker"] + ' ' + str(time(segment["start"])) + '\n'
    output += segment["text"][1:] + ' '
  return output
