import torch
import torchaudio
import pandas as pd

from hyperparams import hp

from text_to_seq import text_to_seq
from mask_from_seq_lengths import mask_from_seq_lengths
from melspecs import convert_to_mel_spec


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.cache = {}

    def get_item(self, row):
      wav_id = row["wav"]                  
      wav_path = f"{hp.wav_path}/{wav_id}.wav"

      text = row["text_norm"]
      text = text_to_seq(text)

      waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
      assert sample_rate == hp.sr

      mel = convert_to_mel_spec(waveform)

      return (text, mel)
    
    def __getitem__(self, index):
      row = self.df.iloc[index]
      wav_id = row["wav"]

      text_mel = self.cache.get(wav_id)

      if text_mel is None:
        text_mel = self.get_item(row)
        self.cache[wav_id] = text_mel
      
      return text_mel

    def __len__(self):
        return len(self.df)


def text_mel_collate_fn(batch):
  text_length_max = torch.tensor(
    [text.shape[-1] for text, _ in batch], 
    dtype=torch.int32
  ).max()

  mel_length_max = torch.tensor(
    [mel.shape[-1] for _, mel in batch],
    dtype=torch.int32
  ).max()

  
  text_lengths = []
  mel_lengths = []
  texts_padded = []
  mels_padded = []

  for text, mel in batch:
    text_length = text.shape[-1]      

    text_padded = torch.nn.functional.pad(
      text,
      pad=[0, text_length_max-text_length],
      value=0
    )

    mel_length = mel.shape[-1]
    mel_padded = torch.nn.functional.pad(
        mel,
        pad=[0, mel_length_max-mel_length],
        value=0
    )

    text_lengths.append(text_length)    
    mel_lengths.append(mel_length)    
    texts_padded.append(text_padded)    
    mels_padded.append(mel_padded)

  text_lengths = torch.tensor(text_lengths, dtype=torch.int32)
  mel_lengths = torch.tensor(mel_lengths, dtype=torch.int32)
  texts_padded = torch.stack(texts_padded, 0)
  mels_padded = torch.stack(mels_padded, 0).transpose(1, 2)

  stop_token_padded = mask_from_seq_lengths(
      mel_lengths,
      mel_length_max
  )
  stop_token_padded = (~stop_token_padded).float()
  stop_token_padded[:, -1] = 1.0
  
  return texts_padded, \
         text_lengths, \
         mels_padded, \
         mel_lengths, \
         stop_token_padded \



if __name__ == "__main__":  
  df = pd.read_csv(hp.csv_path)
  dataset = TextMelDataset(df)

  train_loader = torch.utils.data.DataLoader(
      dataset, 
      num_workers=2, 
      shuffle=True,
      sampler=None, 
      batch_size=hp.batch_size,
      pin_memory=True, 
      drop_last=True, 
      collate_fn=text_mel_collate_fn
  )
  
  def names_shape(names, shape):  
    assert len(names) == len(shape)
    return "(" + ", ".join([f"{k}={v}" for k, v in list(zip(names, shape))]) + ")"

  for i, batch in enumerate(train_loader):
    text_padded, \
    text_lengths, \
    mel_padded, \
    mel_lengths, \
    stop_token_padded = batch

    print(f"=========batch {i}=========")
    print("text_padded:", names_shape(["N", "S"], text_padded.shape))
    print("text_lengths:", names_shape(["N"], text_lengths.shape))
    print("mel_padded:", names_shape(["N", "TIME", "FREQ"], mel_padded.shape))
    print("mel_lengths:", names_shape(["N"], mel_lengths.shape))
    print("stop_token_padded:", names_shape(["N", "TIME"], stop_token_padded.shape))

    if i > 0:
      break

