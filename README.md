# pre-train a custom BERT language model from scratch
Pre-train a costumize BERT with Masked Language Modeling objective on Persian text data using Hugging Face library.
For training on unlabeld text, one great source is https://huggingface.co/datasets/SLPL/naab dataset, contating almost 130 GB of persian unlabeld pre-processed text.
The process of pre-training a language model from scratch involves these main steps:

    Train a Custom Tokenizer on the Text Data:
    Utilize a Word Piece tokenizer specifically designed for BERT models to train on the provided text data.

    Tokenize the Dataset using the Trained Custom Tokenizer:
    Apply the trained custom tokenizer to tokenize the dataset, converting tokens to input_ids.

    Configure and Initialize a Custom BERT Model with Masked Language Modeling Objective:
    Configure and initialize a custom BERT model, focusing on the Masked Language Modeling objective. Alternative objectives may be considered, but it is crucial to ensure compatibility with the appropriate dataset.

    Define Training Arguments and Commence Training:
    Establish the necessary training arguments and initiate the training process.
