from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from pathlib import Path
import os
from wsc_interview import logger
import numpy as np
import torch
import string
import re


# set download root for models
DOWNLOAD_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "models"
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)


def get_bert_uncased_model(cache_dir: Path = DOWNLOAD_ROOT) -> tuple:
    """
    Get bert uncased model
    :return: tokenizer, PT model
    """

    cache_dir = cache_dir / "bert_uncased"
    logger.info("Getting bert uncased model")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              cache_dir=cache_dir,
                                                              output_hidden_states=True)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None, None


def combine_tokens(tokens_embs, tokenized_text, aggregate='max'):
    embs, tokens = [], []
    prev_emb, prev_token = [tokens_embs[0]], tokenized_text[0]
    for idx, (emb, token) in enumerate(zip(tokens_embs[1:], tokenized_text[1:])):
        if token.startswith('##'):
            prev_emb.append(emb)
            prev_token += token[2:]

        if not token.startswith('##') or idx == len(tokenized_text) - 2:
            if aggregate == 'mean':
                embs.append(np.stack(prev_emb).mean(axis=0))
            elif aggregate == 'max':
                embs.append(np.stack(prev_emb).max(axis=0))

            tokens.append(prev_token)
            prev_emb = [emb]
            prev_token = token
    return embs, tokens


def get_bert_embeddings(text, tokenizer, model, aggregate='max'):
    marked_text = f"[CLS] {text} [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # get embeddings
    with torch.no_grad():
        outputs = model(tokens_tensor, torch.ones((1, len(tokenized_text))))
        last_hidden_states = outputs.hidden_states[-1]

    embs, tokens = combine_tokens(last_hidden_states[0], tokenized_text, aggregate=aggregate)
    return embs, tokens


def preprocess_text(text: str, tokenizer=None) -> str:
    """
    Preprocess text data.
    :param text: text to preprocess
    :param tokenizer: tokenizer

    - convert to lower case
    - remove non-ascii characters, punctuation and extra spaces.
    - tokenize the text
    """
    # convert to lower case
    text = text.lower()

    # remove non-ascii characters
    text = text.encode("ascii", "ignore").decode()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    if tokenizer:
        return tokenizer.tokenize(text)
    return text


if __name__ == '__main__':
    tokenizer, model = get_bert_uncased_model()

    text = "here comes called her on the high pick and roll rolling inside lebron james year behind the back floater short"
    phrase = "back floater"
    phrase_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(phrase))

    o = find_tokens(text_tokens, phrase_tokens)
    print(tokenizer)
    print(model)