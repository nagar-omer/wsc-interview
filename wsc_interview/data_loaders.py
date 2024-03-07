import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from wsc_interview import logger
from wsc_interview.bert import preprocess_text, get_bert_uncased_model
from wsc_interview.utils.utils import find_tokens


def find_params(text_tokens: list, prams_tokens: list, params: list) -> dict:
    """
    Check if a parameter is in a given text.
    """
    # parallelize the process
    param_tokens = Parallel(n_jobs=1)(delayed(find_tokens)(text_tokens, phrase_tokens) for phrase_tokens in prams_tokens)
    tokens = [tokens for tokens in param_tokens if tokens]
    params_in_text = [param for tokens, param in zip(param_tokens, params) if tokens]
    return {"Params_tokens": tokens, "Params": params_in_text}


class ActionDataset(Dataset):
    def __init__(self, data_path: str, params_file: str, tokenizer=None):
        self._tokenizer = tokenizer

        self._data = pd.read_csv(data_path)

        # assuming file is coma separated
        self._params = pd.read_csv(params_file)[['parameter']].dropna()

        # preprocess text data (params and data)
        self._params['Tokens'] = self._params['parameter'].apply(lambda x: preprocess_text(x, tokenizer=tokenizer))
        self._data['Tokens'] = self._data['Text'].apply(lambda x: preprocess_text(x, tokenizer=tokenizer))

        # extract parameters from text
        params_df = pd.DataFrame(self._data['Tokens'].apply(lambda x: find_params(x, self._params['Tokens'],
                                                                                  self._params['parameter'])).tolist())
        self._data = pd.concat([self._data, params_df], axis=1)
        self._data['n_params'] = self._data['Params'].apply(len)

        # drop instances with more than one parameter
        self._dropped = self._data[(self._data['n_params'] > 1) | (self._data['n_params'] == 0)]

        logger.info(f"Dropping {len(self._dropped)} instances with zero or more than one parameter")
        self._data = self._data[self._data['n_params'] == 1]
        self._data['Params'] = self._data['Params'].apply(lambda x: x[0])

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
        return self._data.iloc[idx]


if __name__ == '__main__':
    data_path = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/action_enrichment_ds_home_exercise.csv"
    params_file = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/params_list.csv"

    tokenizer, model = get_bert_uncased_model()
    action_dataset = ActionDataset(data_path, params_file, tokenizer=tokenizer)