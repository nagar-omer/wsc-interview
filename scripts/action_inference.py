from wsc_interview.models.action_classification_model import LitActionClassifier
from wsc_interview.models.data_loaders import process_data_and_params
from wsc_interview.models.bert import get_bert_uncased_tokenizer
from pathlib import Path
import pandas as pd
import torch
import yaml
import os


def pipeline(config):
    data_path = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/action_enrichment_ds_home_exercise.csv"
    params_file = config["data"]["params_path"]
    model_path = Path(config["artifacts"]["path"]) / f"{config['model']['name']}_model.pt"

    # load model
    m_dict = torch.load(model_path)
    model = LitActionClassifier(**config)
    model.load_state_dict(m_dict["model"])
    model = model.eval()
    model._threshold = m_dict["threshold"]

    # load data
    df_params = pd.read_csv(params_file)[['parameter']].dropna()
    df_data = pd.read_csv(data_path)[['EventName', 'Text']]
    df_data['instance_id'] = list(range(len(df_data)))

    # preprocess text and params
    tokenizer = get_bert_uncased_tokenizer()
    data, params = process_data_and_params(data=df_data, params=df_params, tokenizer=tokenizer, drop=False, n_jobs=1)

    # prepare data for prediction
    data_wo_params = data[data['n_params'] == 0]
    data_with_params = data[data['n_params'] >= 1]
    data_with_params = data_with_params.explode(['Params', 'Params_tokens'])

    # predict
    data_wo_params['Action'] = 0
    data_with_params['Action'] = data_with_params.apply(lambda instance: predict(model, tokenizer, instance), axis=1)

    # group data
    data = pd.concat([data_wo_params, data_with_params])
    data = data.groupby('instance_id').agg({
        'EventName': 'first',
        'Text': 'first',
        'Tokens': 'first',
        'Params': list,
        'Params_tokens': list,
        'n_params': 'first',
        'Action': list
    }).reset_index()
    data = data.sort_values(by='instance_id')

    # drop instance_id and save
    data = data.drop(columns=['instance_id'])

    artifacts_dir = config["artifacts"]["path"]
    data.to_csv(Path(artifacts_dir) / 'inference.csv', index=False)


def predict(model, tokenizer, instance):
    text_tokens = instance['Tokens']
    parma_tokens = instance['Params_tokens']

    token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    text_token_tensor = tokenizer.prepare_for_model(token_ids, return_tensors='pt')['input_ids']
    return int(model.classifier(text_token_tensor.unsqueeze(dim=0), [parma_tokens]).item() > model.threshold)


if __name__ == '__main__':
    yaml_file = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/config_yamls/baseline_config.yaml"
    assert os.path.exists(yaml_file), f"yaml file {yaml_file} does not exist"
    with open(yaml_file, "r") as f:
        config_ = yaml.safe_load(f)
    pipeline(config_)