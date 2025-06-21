# Code adapted from https://pytorch.org/tutorials/beginner/translation_transformer.html

import matplotlib.pyplot as plt
from model import Seq2SeqTransformer
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import multi30k, Multi30k
from transforms import Transforms
from utils import create_mask 


def train_epoch(model, optimizer, transforms, device):
  model.train()
  losses = 0
  train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=transforms.collate_fn())

  for src, tgt in train_dataloader:
    src = src.to(device)
    tgt = tgt.to(device)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=transforms.PAD_IDX, device=device)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    optimizer.zero_grad()

    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss.backward()

    optimizer.step()
    losses += loss.item()

  return losses / len(list(train_dataloader))


def evaluate(model, transforms, device):
  model.eval()
  losses = 0

  val_iter = Multi30k(split="valid", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=transforms.collate_fn())

  for src, tgt in val_dataloader:
    src = src.to(device)
    tgt = tgt.to(device)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=transforms.PAD_IDX, device=device)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    tgt_out = tgt[1:, :]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    losses += loss.item()
  
  return losses / len(list(val_dataloader))


SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"
transforms = Transforms(SRC_LANGUAGE, TGT_LANGUAGE)
SRC_VOCAB_SIZE = len(transforms.vocab()[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(transforms.vocab()[TGT_LANGUAGE])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# Init model
transformer = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, emb_size=EMB_SIZE, nhead=NHEAD, src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE)
for p in transformer.parameters():
  if p.dim() > 1:
    nn.init.xavier_uniform_(p)
transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=transforms.PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Train
train_losses = []
val_losses = []
with open("training.log", "w") as f:
  for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, transforms, DEVICE)
    end_time = timer()
    val_loss = evaluate(transformer, transforms, DEVICE)

    # Log epoch stats
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    epoch_stats = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time: {(end_time - start_time):.3f}s"
    print(epoch_stats)
    print(epoch_stats, file=f)

# Save model
torch.save(transformer.state_dict(), "Seq2SeqTransformer")

# Plot losses
xs = list(torch.arange(1, NUM_EPOCHS+1))
fig, ax = plt.subplots()
ax.plot(xs, train_losses)
ax.plot(xs, val_losses)
ax.set_title("Train/Val Loss")
ax.set_ylabel("loss")
ax.set_xlabel("epoch")
ax.legend(["train", "val"], loc="upper right")
fig.savefig("train_val_loss.png")