# Code adapted from https://pytorch.org/tutorials/beginner/translation_transformer.html

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List

# Helper methods for sequence transformation
class Transforms:
  def __init__(self, src_lang, tgt_lang):
    self.SPECIAL_SYMBOLS = ["<unk>", "<pad>", "<bos>", "<eos>"]
    self.UNK_IDX = 0
    self.PAD_IDX = 1
    self.BOS_IDX = 2
    self.EOS_IDX = 3

    self.src_lang = src_lang
    self.tgt_lang = tgt_lang

    self.token_transform = {}
    self.vocab_transform = {}
    self.text_transform = {}

    # Get tokenizer
    if src_lang == "de":
      self.token_transform[self.src_lang] = get_tokenizer("spacy", language="de_core_news_sm")
      self.token_transform[self.tgt_lang] = get_tokenizer("spacy", language="en_core_web_sm")
    else:
      self.token_transform[self.src_lang] = get_tokenizer("spacy", language="en_core_web_sm")
      self.token_transform[self.tgt_lang] = get_tokenizer("spacy", language="de_core_news_sm")

    # Wrap the dataset iterable in an iterable that generates tokens from tokenizer
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
      language_index = {self.src_lang: 0, self.tgt_lang: 1}
      for data_sample in data_iter:
        yield self.token_transform[language](data_sample[language_index[language]])

    # Chain transforms
    def sequential_transforms(*transforms):
      def func(txt_input):
        for transform in transforms:
          txt_input = transform(txt_input)
        return txt_input
      return func

    for ln in [self.src_lang, self.tgt_lang]:
      # Build vocab
      train_iter = Multi30k(split="train", language_pair=(self.src_lang, self.tgt_lang))
      self.vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln), min_freq=1, specials=self.SPECIAL_SYMBOLS, special_first=True)
      self.vocab_transform[ln].set_default_index(self.UNK_IDX)

      # Get text transform
      self.text_transform[ln] = sequential_transforms(self.token_transform[ln], self.vocab_transform[ln], self.tensor)

  # Provide the tokenizer transforms
  def token(self):
    return self.token_transform

  # Provide the vocab index transforms
  def vocab(self):
    return self.vocab_transform

  # Provide the BOS prepend and EOS append transform
  def tensor(self, token_ids: List[int]):
    return torch.cat((torch.tensor([self.BOS_IDX]), torch.tensor(token_ids), torch.tensor([self.EOS_IDX])))

  # Provide the aggregate end-to-end transforms
  def text(self):
    return self.text_transform
  
  # Provide a collation function for training
  def collate_fn(self):
    def c_fn(batch):
      src_batch, tgt_batch = [], []
      for src_sample, tgt_sample in batch:
        src_batch.append(self.text_transform[self.src_lang](src_sample.rstrip("\n")))
        tgt_batch.append(self.text_transform[self.tgt_lang](tgt_sample.rstrip("\n")))
      
      src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
      tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
      return src_batch, tgt_batch
    return c_fn
