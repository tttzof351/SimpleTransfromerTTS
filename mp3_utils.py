import numpy as np
import pydub
from hyperparams import hp

def write_mp3(
  x, 
  f="audio.mp3", 
  sr=hp.sr, 
  normalized=True
):
  """numpy array to MP3"""
  channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
  if normalized:  # normalized array - each item should be a float in [-1, 1)
      y = np.int16(x * 2 ** 15)
  else:
      y = np.int16(x)
  song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
  song.export(f, format="mp3", bitrate="320k")