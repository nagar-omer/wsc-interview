from wsc_interview.models.action_classification_model import LitActionClassifier
from wsc_interview.models.data_loaders import ActionDataset, CollateFn
from wsc_interview.models.bert import get_bert_uncased_tokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import yaml
import os


def pipeline(config, data_path):
    params_file = config["data"]["params_path"]
    model_path = Path(config["artifacts"]["path"]) / f"{config['model']['name']}_model.pt"
    use_mask = config["data"]["use_mask"]
    batch_size = config["inference"]["batch_size"]

    # load model
    m_dict = torch.load(model_path)
    model = LitActionClassifier(**config)
    model.load_state_dict(m_dict["model"])
    model = model.eval()
    model._threshold = m_dict["threshold"]

    # load data
    tokenizer = get_bert_uncased_tokenizer()
    ds = ActionDataset(data_path, params_file, tokenizer=tokenizer, use_mask=use_mask, mode='inference')
    dl = DataLoader(ds, num_workers=8, batch_size=batch_size, shuffle=False,
                    collate_fn=CollateFn(mode='inference'), persistent_workers=True)

    # predict
    all_text = ds.dropped_instances['Text'].tolist()
    all_params = [None] * len(all_text)
    all_labels = [0] * len(all_text)
    for token_ids, phrase_token_idx, text, phrase in tqdm(dl, total=len(dl)):
        out = model.classifier(token_ids, phrase_token_idx).detach().cpu()
        all_text.extend(text)
        all_params.extend(phrase)
        all_labels.extend((out > model.threshold).squeeze(dim=1).numpy().astype(np.uint8).tolist())

    # save predictions
    res_df = pd.DataFrame({"Text": all_text, "Params": all_params, "Label": all_labels})
    res_path = Path(config["artifacts"]["path"]) / f"inference.csv"
    res_df.to_csv(res_path, index=False)


if __name__ == '__main__':
    data_path = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/data/action_enrichment_ds_home_exercise.csv"
    yaml_file = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/config_yamls/baseline_config.yaml"
    assert os.path.exists(yaml_file), f"yaml file {yaml_file} does not exist"
    with open(yaml_file, "r") as f:
        config_ = yaml.safe_load(f)
    pipeline(config_, data_path=data_path)