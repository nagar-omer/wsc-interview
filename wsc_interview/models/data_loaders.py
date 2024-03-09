from wsc_interview.models.bert import preprocess_text, get_bert_uncased_tokenizer
from wsc_interview.utils.utils import find_tokens
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from wsc_interview import logger
from collections import Counter
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import os


def find_params(text_tokens: list, prams_tokens: list, params: list, n_jobs=-1) -> dict:
    """
    Check if a parameter is in a given text.
    """
    # parallelize the process
    param_tokens = Parallel(n_jobs=n_jobs)(delayed(find_tokens)(text_tokens, phrase_tokens) for phrase_tokens in prams_tokens)
    tokens = [tokens for tokens in param_tokens if tokens]
    params_in_text = [param for tokens, param in zip(param_tokens, params) if tokens]
    return {"Params_tokens": tokens, "Params": params_in_text}


def process_data_and_params(data: pd.DataFrame, params: pd.DataFrame, tokenizer, drop=True, n_jobs=-1) -> Tuple:

    # preprocess text data (params and data)
    params['Tokens'] = params['parameter'].apply(lambda x: preprocess_text(x, tokenizer=tokenizer))
    data['Tokens'] = data['Text'].apply(lambda x: preprocess_text(x, tokenizer=tokenizer))

    # extract parameters from text
    params_df = pd.DataFrame(data['Tokens'].apply(lambda x: find_params(x, params['Tokens'],
                                                                           params['parameter'],
                                                                        n_jobs=n_jobs)).tolist())
    data = pd.concat([data, params_df], axis=1)
    data['n_params'] = data['Params'].apply(len)

    if not drop:
        return data, params

    # drop instances with more than one parameter
    dropped = data[(data['n_params'] > 1) | (data['n_params'] == 0)]

    logger.info(f"Dropping {len(dropped)} instances with zero or more than one parameter")
    data = data[data['n_params'] == 1]
    data['Params'] = data['Params'].apply(lambda x: x[0])
    data['Params_tokens'] = data['Params_tokens'].apply(lambda x: x[0])

    return data, params, dropped


class ActionDataset(Dataset):
    def __init__(self, data_path: str = None, params_file: str = None, tokenizer=None):

        if not os.path.exists(data_path):
            logger.error(f"Data file {data_path} does not exist")
            return

        if not os.path.exists(params_file):
            logger.error(f"Params file {params_file} does not exist")
            return

        self._tokenizer = tokenizer if tokenizer else get_bert_uncased_tokenizer()
        self._data = pd.read_csv(data_path)
        # assuming file is coma separated
        self._params = pd.read_csv(params_file)[['parameter']].dropna()
        self._data, self._params, self._dropped = process_data_and_params(self._data, self._params, self._tokenizer)

    @property
    def dropped_instances(self):
        return self._dropped

    @property
    def all_instances(self):
        transcriptions = self._data['Text']
        params = self._data['Params']
        labels = self._data['Label']
        return transcriptions, params, labels

    @property
    def params_classes(self):
        return self._params['parameter'].values

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # get tokens and labels
        tokens = self._data.iloc[idx]['Tokens']

        # convert tokens to ids and prepare for model
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self._tokenizer.prepare_for_model(token_ids, return_tensors='pt')['input_ids']

        # get phrase tokens
        phrase_token_idx = torch.Tensor(self._data.iloc[idx]['Params_tokens']).long()
        label = int(self._data.iloc[idx]['Label'])

        return token_ids, phrase_token_idx, label

    def split(self, train_size: float = 0.8):
        """
        Split data to train and test
        """

        # get distribution of actions and labels
        labels = self._data['Label'].tolist()
        actions = self._data['Params'].tolist()
        action_labels_freq = Counter(list(zip(actions, labels)))

        train_loc, test_loc = [], []
        for (action, label), freq in action_labels_freq.items():
            # get location of each pair and shuffle
            pair_loc = self._data[(self._data['Params'] == action) & (self._data['Label'] == label)].index.tolist()
            pair_loc = np.random.permutation(pair_loc)

            # split
            train_loc.extend(pair_loc[:int(freq * train_size)])
            test_loc.extend(pair_loc[int(freq * train_size):])

        # duplicate dataset
        train_data = deepcopy(self)
        test_data = deepcopy(self)

        # filter data based on location
        train_data._data = train_data._data.loc[train_loc]
        test_data._data = test_data._data.loc[test_loc]
        return train_data, test_data


def collate_fn(batch):
    """
    Collate function for the dataloader
    """
    # extract tokens, phrase_token_idx and labels
    tokens, phrase_token_idx, labels = zip(*batch)

    # pad tokens with [PAD] - 0
    max_len = max([len(t) for t in tokens])
    tokens = [torch.cat((t, torch.zeros(max_len - len(t)).long())) for t in tokens]
    tokens = torch.stack(tokens)

    # keep phrase as list
    phrase_token_idx = phrase_token_idx

    # to tensor and long
    labels = torch.Tensor(labels).long()
    return tokens, phrase_token_idx, labels


if __name__ == '__main__':
    data_path = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/action_enrichment_ds_home_exercise.csv"
    params_file = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/params_list.csv"

    tokenizer = get_bert_uncased_tokenizer()

    action_dataset = ActionDataset(data_path, params_file, tokenizer=tokenizer)
    train_ds, test_ds = action_dataset.split()