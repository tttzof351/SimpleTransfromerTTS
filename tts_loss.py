from hyperparams import hp
import torch

class TTSLoss(torch.nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/loss_function.py"""
    def __init__(self):
        super(TTSLoss, self).__init__()
        
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, 
        mel_postnet_out, 
        mel_out, 
        stop_token_out, 
        mel_target, 
        stop_token_target
      ):      
        stop_token_target = stop_token_target.view(-1, 1)

        stop_token_out = stop_token_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + \
            self.mse_loss(mel_postnet_out, mel_target)

        stop_token_loss = self.bce_loss(stop_token_out, stop_token_target) * hp.r_gate

        return mel_loss + stop_token_loss

if __name__ == "__main__":
  loss = TTSLoss()
  print(loss)  
