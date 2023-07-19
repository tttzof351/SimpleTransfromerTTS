import os 

import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from hyperparams import hp
from dataset import TextMelDataset, text_mel_collate_fn
from tts_loss import TTSLoss
from model import TransformerTTS
from melspecs import inverse_mel_spec_to_wav
from text_to_seq import text_to_seq


def batch_process(batch):
  text_padded, \
  text_lengths, \
  mel_padded, \
  mel_lengths, \
  stop_token_padded = batch

  text_padded = text_padded.cuda()
  text_lengths = text_lengths.cuda()
  mel_padded = mel_padded.cuda()
  stop_token_padded = stop_token_padded.cuda()
  mel_lengths = mel_lengths.cuda()

  N = mel_padded.shape[0]
  SOS = torch.zeros((N, 1, hp.mel_freq), device=mel_padded.device) # Start of sequence
  
  mel_input = torch.cat(
    [
      SOS, 
      mel_padded[:, :-1, :] # (N, L, FREQ)
    ],
    dim=1
  )  

  return text_padded, \
         text_lengths, \
         mel_padded, \
         mel_lengths, \
         mel_input, \
         stop_token_padded



def inference_utterance(model, text):
  sequences = text_to_seq(text).unsqueeze(0).cuda()
  postnet_mel, stop_token = model.inference(
    sequences, 
    stop_token_threshold=1e5, 
    with_tqdm = False
  )          
  audio = inverse_mel_spec_to_wav(postnet_mel.detach()[0].T)
            
  fig, (ax1) = plt.subplots(1, 1)
  ax1.imshow(
      postnet_mel[0, :, :].detach().cpu().numpy().T, 
  )
  
  return audio, fig 


def calculate_test_loss(model, test_loader):
  test_loss_mean = 0.0
  model.eval()

  with torch.no_grad():
    for test_i, test_batch in enumerate(test_loader):
      test_text_padded, \
      test_text_lengths, \
      test_mel_padded, \
      test_mel_lengths, \
      test_mel_input, \
      test_stop_token_padded = batch_process(batch)

      test_post_mel_out, test_mel_out, test_stop_token_out = model(
        test_text_padded, 
        test_text_lengths,
        test_mel_input, 
        test_mel_lengths
      )        
      test_loss = criterion(
        mel_postnet_out = test_post_mel_out, 
        mel_out = test_mel_out, 
        stop_token_out = test_stop_token_out, 
        mel_target = test_mel_padded, 
        stop_token_target = test_stop_token_padded
      )

      test_loss_mean += test_loss.item()

  test_loss_mean = test_loss_mean / (test_i + 1)  
  return test_loss_mean


if __name__ == "__main__":
  torch.manual_seed(hp.seed)

  df = pd.read_csv(hp.csv_path)  
  train_df, test_df = train_test_split(
    df, 
    test_size=64, 
    random_state=hp.seed
  )
  train_loader = torch.utils.data.DataLoader(
      TextMelDataset(train_df), 
      num_workers=2, 
      shuffle=True,
      sampler=None, 
      batch_size=hp.batch_size,
      pin_memory=True, 
      drop_last=True, 
      collate_fn=text_mel_collate_fn
  )
  test_loader = torch.utils.data.DataLoader(
      TextMelDataset(test_df), 
      num_workers=2, 
      shuffle=True,
      sampler=None, 
      batch_size=8,
      pin_memory=True, 
      drop_last=True, 
      collate_fn=text_mel_collate_fn
  )  
  
  train_saved_path = f"{hp.save_path}/train_{hp.save_name}"
  test_saved_path = f"{hp.save_path}/test_{hp.save_name}"
  
  print("train_saved_path:", train_saved_path)
  print("test_saved_path:", test_saved_path)

  logger = SummaryWriter(hp.log_path)  
  criterion = TTSLoss().cuda()
  model = TransformerTTS().cuda()
  optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr)
  scaler = torch.cuda.amp.GradScaler()  

  best_test_loss_mean = float("inf")
  best_train_loss_mean = float("inf")
  
  train_loss_mean = 0.0
  epoch = 0
  i = 0

  if os.path.isfile(train_saved_path):  
    state = torch.load(train_saved_path)
    state_model = state["model"]
    state_optimizer = state["optimizer"]
    
    i = state["i"] + 1
    best_test_loss_mean = state.get("test_loss", float("inf"))
    best_train_loss_mean = state.get("train_loss", float("inf"))

    model.load_state_dict(state_model)
    optimizer.load_state_dict(state_optimizer)

    print(f"Load: {i}; test_loss: {np.round(best_test_loss_mean, 5)}; train_loss: {np.round(best_train_loss_mean, 5)}")
  else:
    print("Start from zero!")


  start_time_sec = time.time()
  while True:
    for batch in train_loader:      
      text_padded, \
      text_lengths, \
      mel_padded, \
      mel_lengths, \
      mel_input, \
      stop_token_padded = batch_process(batch)

      model.train(True)
      model.zero_grad()

      with torch.autocast(device_type='cuda', dtype=torch.float16):
        post_mel_out, mel_out, stop_token_out = model(
          text_padded, 
          text_lengths,
          mel_input, 
          mel_lengths
        )        
        loss = criterion(
          mel_postnet_out = post_mel_out, 
          mel_out = mel_out, 
          stop_token_out = stop_token_out, 
          mel_target = mel_padded, 
          stop_token_target = stop_token_padded
        )

      scaler.scale(loss).backward()      
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
      scaler.step(optimizer)
      scaler.update()

      train_loss_mean += loss.item()      

      if i !=0 and i % hp.step_print == 0:
        train_loss_mean = train_loss_mean / hp.step_print        
        logger.add_scalar("Loss/train_loss", train_loss_mean, global_step=i)
        
        if i % hp.step_test == 0:            
          test_loss_mean = calculate_test_loss(model, test_loader)
          audio, fig = inference_utterance(model, "Hello, World.")

          logger.add_scalar("Loss/test_loss", test_loss_mean, global_step=i)
          logger.add_figure(f"Img/img_{i}", fig, global_step=i) 
          logger.add_audio(f"Utterance/audio_{i}",audio, sample_rate=hp.sr, global_step=i)
          
          print(f"{epoch}-{i}) Test loss: {np.round(test_loss_mean, 5)}")

          if i % hp.step_save == 0:
            is_best_train = train_loss_mean < best_train_loss_mean
            is_best_test = test_loss_mean < best_test_loss_mean

            state = {
              "model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "i": i,
              "test_loss": test_loss_mean,
              "train_loss": train_loss_mean
            }

            if is_best_train:
              print(f"{epoch}-{i}) Save best train")
              torch.save(state, train_saved_path)
              best_train_loss_mean = train_loss_mean

            if is_best_test:
              print(f"{epoch}-{i}) Save best test")
              torch.save(state, test_saved_path)
              best_test_loss_mean = test_loss_mean
              

        end_time_sec = time.time()
        time_sec = np.round(end_time_sec - start_time_sec, 3)
        start_time_sec = end_time_sec
        
        print(f"{epoch}-{i}) Train loss: {np.round(train_loss_mean, 5)}; Duration: {time_sec} sec.")
        train_loss_mean = 0.0

      i += 1
    epoch += 1      
