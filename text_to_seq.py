import torch
from hyperparams import hp

symbol_to_id = {
  s: i for i, s in enumerate(hp.symbols)
}

def text_to_seq(text):
  text = text.lower()
  seq = []
  for s in text:
    _id = symbol_to_id.get(s, None)
    if _id is not None:
      seq.append(_id)

  seq.append(symbol_to_id["EOS"])

  return torch.IntTensor(seq)


if __name__ == "__main__":
  print(text_to_seq("Hello, World"))