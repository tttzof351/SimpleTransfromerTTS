from hyperparams import hp
import torch
import torchaudio
from torchaudio.functional import spectrogram


spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=hp.n_fft, 
    win_length=hp.win_length,
    hop_length=hp.hop_length,
    power=hp.power
)


mel_scale_transform = torchaudio.transforms.MelScale(
  n_mels=hp.mel_freq, 
  sample_rate=hp.sr, 
  n_stft=hp.n_stft
)


mel_inverse_transform = torchaudio.transforms.InverseMelScale(
  n_mels=hp.mel_freq, 
  sample_rate=hp.sr, 
  n_stft=hp.n_stft
).cuda()


griffnlim_transform = torchaudio.transforms.GriffinLim(
    n_fft=hp.n_fft,
    win_length=hp.win_length,
    hop_length=hp.hop_length
).cuda()


def norm_mel_spec_db(mel_spec):  
  mel_spec = ((2.0*mel_spec - hp.min_level_db) / (hp.max_db/hp.norm_db)) - 1.0
  mel_spec = torch.clip(mel_spec, -hp.ref*hp.norm_db, hp.ref*hp.norm_db)
  return mel_spec


def denorm_mel_spec_db(mel_spec):
  mel_spec = (((1.0 + mel_spec) * (hp.max_db/hp.norm_db)) + hp.min_level_db) / 2.0 
  return mel_spec


def pow_to_db_mel_spec(mel_spec):
  mel_spec = torchaudio.functional.amplitude_to_DB(
    mel_spec,
    multiplier = hp.ampl_multiplier, 
    amin = hp.ampl_amin, 
    db_multiplier = hp.db_multiplier, 
    top_db = hp.max_db
  )
  mel_spec = mel_spec/hp.scale_db
  return mel_spec


def db_to_power_mel_spec(mel_spec):
  mel_spec = mel_spec*hp.scale_db
  mel_spec = torchaudio.functional.DB_to_amplitude(
    mel_spec,
    ref=hp.ampl_ref,
    power=hp.ampl_power
  )  
  return mel_spec


def convert_to_mel_spec(wav):
  spec = spec_transform(wav)
  mel_spec = mel_scale_transform(spec)
  db_mel_spec = pow_to_db_mel_spec(mel_spec)
  db_mel_spec = db_mel_spec.squeeze(0)
  return db_mel_spec


def inverse_mel_spec_to_wav(mel_spec):
  power_mel_spec = db_to_power_mel_spec(mel_spec)
  spectrogram = mel_inverse_transform(power_mel_spec)
  pseudo_wav = griffnlim_transform(spectrogram)
  return pseudo_wav


if __name__ == "__main__":
  wav_path = f"{hp.wav_path}/LJ023-0073.wav" 
  waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
  mel_spec = convert_to_mel_spec(waveform)
  print("mel_spec:", mel_spec.shape)

  pseudo_wav = inverse_mel_spec_to_wav(mel_spec.cuda())
  print("pseudo_wav:", pseudo_wav.shape)