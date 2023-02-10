from datasets import *
from transformers import *
from tokenizers import BertWordPieceTokenizer
from datasets import load_dataset
import os
import json

#LOADING DATASET
#------------------------------------------------------------------------------------------------------------------------------------
# You should just change this part in order to download your
# parts of corpus. there are 126 training files and 3 test files 
# all in .text format, the indices start from 0
def data_files_dir(data_path : str ,train_file_idx : list , test_file_idx : list ):
    """
    This function gets a path to the local dataset and returns a list of files 
    the naming of files is compatible with the dataset namings originaly made by the authors
    if you have your dataset downloaded use the specified directory or you can use the link to the dataset
    "https://huggingface.co/datasets/SLPL/naab/resolve/main/data/"

    Args: 
        data_path: path to the dataset 
        train_file_idx: which list of indexes of the train data to choose
        test_file_idx: which list of indexes of the test data to choose 
    
    Returns:
        A dictionary of train and test files directories
    """
    indices = {
        "train": train_file_idx ,
        "test" : test_file_idx
    }
    N_FILES = {
        "train": 126,
        "test": 3
    }
    _BASE_URL = data_path + "/"
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
    return data_files
    
data_files =  data_files_dir(data_path = "D:/Thesis_Project/dataset",train_file_idx = [0],test_file_idx = [2])
print(data_files)
#--------------------------------------------------------------------------------------------------------------------------------------

#CONVERTING DATASET OBJECT TO FILES
#--------------------------------------------------------------------------------------------------------------------------------------
# def dataset_to_text(dataset, output_filename="data.txt"):

#   """ Utility function to save dataset text to disk,
#   useful for using the texts to train the tokenizer 
#   (as the tokenizer accepts files) """

#   with open(output_filename, "w") as f:
#     for t in dataset["text"]:
#       print(t, file=f)

# # save the training set to train.txt
# dataset_to_text(dataset["train"], "train.txt")
# # save the testing set to test.txt
# dataset_to_text(dataset["test"], "test.txt")
#--------------------------------------------------------------------------------------------------------------------------------------


# DEFINING TOKENIZER'S PARAMETERES 
#--------------------------------------------------------------------------------------------------------------------------------------
special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# training the tokenizer on the training set
files = data_files["train"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512
# whether to truncate
truncate_longer_samples = False
#--------------------------------------------------------------------------------------------------------------------------------------

# TRAINING THE TOKENIZER
#--------------------------------------------------------------------------------------------------------------------------------------
# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size,
                special_tokens=special_tokens)

# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)
#--------------------------------------------------------------------------------------------------------------------------------------

# SAVING THE TOKENIZER
#--------------------------------------------------------------------------------------------------------------------------------------
model_path = "pretrained_bert_tokenizer"

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
#--------------------------------------------------------------------------------------------------------------------------------------

#LOADING THE TOKENIZER AS BertTokenizerFast
#--------------------------------------------------------------------------------------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained(model_path)
#--------------------------------------------------------------------------------------------------------------------------------------

#TOEKNIZING THE DATASET
#--------------------------------------------------------------------------------------------------------------------------------------

# Mapping function to tokenize the sentences passed with truncation
def encode_with_truncation(examples):
  return tokenizer(examples["text"], truncation=True, padding="max_length",
                   max_length=max_length, return_special_tokens_mask=True)

# Mapping function to tokenize the sentences passed without truncation
def encode_without_truncation(examples):
  return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# creating a Dataset object so that we can use the map() method 
cache_dir = "D:/Thesis_Project/cache_dir"
dataset = load_dataset('text', data_files=data_files,
                        cache_dir = cache_dir,
                        use_auth_token=True)

# tokenizing the train dataset
train_dataset = dataset["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = dataset["test"].map(encode, batched=True)

# SAVING TOKENIZED DATASET
tokenized_dataset_path = "tokenized_dataset"

if not os.path.isdir(tokenized_dataset_path):
    os.mkdir(tokenized_dataset_path)

train_dataset.save_to_disk(tokenized_dataset_path)
test_dataset.save_to_disk(tokenized_dataset_path)

if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as PyTorch tensors
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  # remove other columns, and remain them as Python lists
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
#--------------------------------------------------------------------------------------------------------------------------------------

# in the case of setting truncate_longer_samples to False, 
# we need to join our untruncated samples together and cut them into fixed-size vectors,
# since the model expects a fixed-sized sequence during training.
#--------------------------------------------------------------------------------------------------------------------------------------
from itertools import chain
# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
# grabbed from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
# remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
# might be slower to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
if not truncate_longer_samples:
  train_dataset = train_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
  test_dataset = test_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
  # convert them from lists to torch tensors
  train_dataset.set_format("torch")
  test_dataset.set_format("torch")
#--------------------------------------------------------------------------------------------------------------------------------------

"""If you don't want to concatenate all texts and then split them into chunks of 512 tokens,
then make sure you set truncate_longer_samples to True,
so it will treat each line as an individual sample regardless of its length.
If you set truncate_longer_samples to True, the above code cell won't be executed at all."""

# LOADING THE MODEL
#--------------------------------------------------------------------------------------------------------------------------------------
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)
#--------------------------------------------------------------------------------------------------------------------------------------

# TRAINING THE MODEL
#--------------------------------------------------------------------------------------------------------------------------------------
# initialize the data collator,
# randomly masking 20% (default is 15%) of the tokens
# for the Masked Language Modeling (MLM) task

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
#--------------------------------------------------------------------------------------------------------------------------------------

# INITIALIZE THE TRAINING ARFUMENTS
#--------------------------------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=10,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    load_best_model_at_end=True,    # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)
#--------------------------------------------------------------------------------------------------------------------------------------

#INITIALIZE THE TRAINER 
#--------------------------------------------------------------------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
#--------------------------------------------------------------------------------------------------------------------------------------

# TRAIN THE MODEL!!!
#--------------------------------------------------------------------------------------------------------------------------------------
trainer.train()
#--------------------------------------------------------------------------------------------------------------------------------------
