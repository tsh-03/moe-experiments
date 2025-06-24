# Mixture of Experts (MoE) Transformer with Llama4 type model
# Author: Tirth Shah
# Inspired by: https://github.com/FareedKhan-dev/train-llama4 (MIT License)
# Inference script

# ------------------------------------------------------------------------------

import torch
from model import MoETransformer, MoEConfig
from prepare_data import CharDataset

text = open("tiny_shakespeare.txt", encoding="utf-8").read()
dataset = CharDataset(text)
config = MoEConfig()
model = MoETransformer(config, dataset.vocab_size)
model.load_state_dict(torch.load("moe_transformer.pth"))
model.eval()

context = "ROMEO:"
input_ids = torch.tensor([[dataset.stoi[c] for c in context]], dtype=torch.long)
with torch.no_grad():
    logits = model(input_ids)
    next_id = torch.argmax(logits[0, -1]).item()
    print("Next character prediction:", dataset.itos[next_id])
