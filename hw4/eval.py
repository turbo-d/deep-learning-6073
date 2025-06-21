# Code adapted from https://pytorch.org/tutorials/beginner/translation_transformer.html

from model import Seq2SeqTransformer
import torch
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torchtext.datasets import multi30k, Multi30k
from transforms import Transforms
from utils import generate_square_subsequent_mask 


def greedy_decode(model, transforms, src, src_mask, max_len, start_symbol, device):
  src = src.to(device)
  src_mask = src_mask.to(device)

  memory = model.encode(src, src_mask)
  ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
  for i in range(max_len-1):
    memory = memory.to(device)
    tgt_mask = (generate_square_subsequent_mask(ys.size(0), device=device).type(torch.bool)).to(device)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()

    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    if next_word == transforms.EOS_IDX:
      break

  return ys


def translate(model: torch.nn.Module, transforms: Transforms, src_sentence: str, src_lang: str, tgt_lang: str, device: str):
  model.eval()
  src = transforms.text()[src_lang](src_sentence).view(-1, 1)
  num_tokens = src.shape[0]
  src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
  tgt_tokens = greedy_decode(model, transforms, src, src_mask, max_len=num_tokens + 5, start_symbol=transforms.BOS_IDX, device=device).flatten()
  tgt_words = transforms.vocab()[tgt_lang].lookup_tokens(list(tgt_tokens.cpu().numpy()))
  bos_word = transforms.SPECIAL_SYMBOLS[transforms.BOS_IDX]
  eos_word = transforms.SPECIAL_SYMBOLS[transforms.EOS_IDX]
  return " ".join(tgt_words).replace(bos_word, "").replace(eos_word, "").strip()


def test(model, transforms, device):
  model.eval()

  test_iter = Multi30k(split="test", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  test_dataloader = DataLoader(test_iter, batch_size=None)

  candidate_corpus = []
  references_corpus = []
  for src, tgt in test_dataloader:
    candidate = translate(model, transforms, src, SRC_LANGUAGE, TGT_LANGUAGE, device)
    candidate_corpus.append(candidate.split(" "))
    references_corpus.append([tgt.split(" ")])
  
  return bleu_score(candidate_corpus, references_corpus)



SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"
transforms = Transforms(SRC_LANGUAGE, TGT_LANGUAGE)
SRC_VOCAB_SIZE = len(transforms.vocab()[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(transforms.vocab()[TGT_LANGUAGE])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# Load trained model
transformer = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, emb_size=EMB_SIZE, nhead=NHEAD, src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE)
transformer.load_state_dict(torch.load("Seq2SeqTransformer"))
transformer.to(DEVICE)

# Eval metrics
print(f"BLEU score: {test(transformer, transforms, DEVICE):.5f}")

# Translation example
input =  "Ein Mann steht auf einem felsigen Hügel"
target = "A man standing on a rocky hill"
prediction = translate(transformer, transforms, input, SRC_LANGUAGE, TGT_LANGUAGE, DEVICE)
print("Example:")
print(f"input: {input}")
print(f"prediction: {prediction}")
print(f"target: {target}")