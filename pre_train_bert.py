from datasets import *
from transformers import *
from tokenizers import *
from tokenizers import BertWordPieceTokenizer
import os
import json

from datasets import load_dataset

# ==========================================================
# You should just change this part in order to download your 
# parts of corpus.

indices = {
    "train": [0],
    "test": [2]
}
# ==========================================================


N_FILES = {
    "train": 126,
    "test": 3
}
_BASE_URL = "https://huggingface.co/datasets/SLPL/naab/resolve/main/data/"
data_url = {
    "train": [_BASE_URL + "train-{:05d}-of-{:05d}.txt".format(x, N_FILES["train"]) for x in range(N_FILES["train"])],
    "test": [_BASE_URL + "test-{:05d}-of-{:05d}.txt".format(x, N_FILES["test"]) for x in range(N_FILES["test"])],
}
for index in indices['train']:
    assert index < N_FILES['train']
for index in indices['test']:
    assert index < N_FILES['test']
data_files = {
    "train": [data_url['train'][i] for i in indices['train']],
    "test": [data_url['test'][i] for i in indices['test']]
}
print(data_files)
dataset = load_dataset('text', data_files=data_files, use_auth_token=True)

# since we our files in text format
# defining some parameters

special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = ["train.txt"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512
# whether to truncate
truncate_longer_samples = False

# training the tokenizer 
# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

# saveing the tokenizer
model_path = "pretrained_bert"
# make the directory if not already there
if not os.path.isdir(model_path):
  os.mkdir(model_path)

# save the tokenizer  
tokenizer.save_model(model_path)

# dumping some of the tokenizer config to config file, 
# including special tokens, whether
# to lower case and the maximum sequence length

with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)