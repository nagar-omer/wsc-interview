from wsc_interview.models.action_classification_model import LitActionClassifier
from wsc_interview.models.data_loaders import ActionDataset, Collate_fn
from wsc_interview.models.bert import get_bert_uncased_tokenizer
from torch.utils.data import DataLoader
from pathlib import Path
import lightning as L
import numpy as np
import torch
import yaml
import os


def plot_loss(train_loss, val_loss, filename=None):
    import matplotlib.pyplot as plt
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("CE-Loss")
    plt.title("Loss over epochs")
    plt.legend()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def action_phrase_accuracy(model, ds):
    """
    Calculate accuracy for action phrase
    """
    dl = DataLoader(ds, num_workers=1, batch_size=1, shuffle=False, collate_fn=Collate_fn, persistent_workers=True)
    # get predictions
    phrases, predictions, labels = [], [], []
    for batch in dl:
        tokens, phrase_token_idx, phrase, label = batch
        predictions.append(model.classifier(tokens, phrase_token_idx).detach().numpy().item())
        phrases.append(phrase[0])
        labels.append(label.item())

    # calculate accuracy for each action
    accuracy_dict = {}
    for action in set(phrases):
        action_preds = [p for p, a in zip(predictions, phrases) if a == action]
        action_preds = (np.array(action_preds) > model.threshold).astype(int)
        action_labels = [l for l, a in zip(labels, phrases) if a == action]

        acc = (action_preds == action_labels).mean()
        accuracy_dict[action] = acc

    return accuracy_dict


def train(config: dict):
    # data params
    data_path = config["data"]["data_path"]
    params_file = config["data"]["params_path"]
    train_size = config["data"]["train_val_split"]
    use_mask = config["data"]["use_mask"]

    # training params
    bach_size = config["training"]["batch_size"]
    max_epochs = config["training"]["max_epochs"]
    num_workers = config["training"]["num_workers"]

    # set date loaders
    tokenizer = get_bert_uncased_tokenizer()
    action_dataset = ActionDataset(data_path, params_file, tokenizer=tokenizer, use_mask=use_mask)
    train_ds, test_ds = action_dataset.split(train_size=train_size)

    dl_train = DataLoader(train_ds, num_workers=num_workers, batch_size=bach_size,
                          shuffle=True, collate_fn=Collate_fn, persistent_workers=True)
    dl_eval = DataLoader(test_ds, num_workers=num_workers, batch_size=num_workers,
                         shuffle=False, collate_fn=Collate_fn, persistent_workers=True)

    # load model
    model = LitActionClassifier(label_count=action_dataset.all_instances[2].value_counts(), **config)

    # train model
    trainer = L.Trainer(max_epochs=max_epochs, accelerator='cpu')
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_eval)

    # calculate accuracy per action
    action_accuracy = action_phrase_accuracy(model, test_ds)
    print(action_accuracy)

    # save model & threshold
    name = config["model"]["name"]
    save_dir = Path(config["artifacts"]["path"])
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "threshold": model._threshold}, save_dir / f"{name}_model.pt")

    # plot loss
    plot_loss(model._train_loss, model._val_loss, filename=save_dir / f"{name}_loss.png")


if __name__ == '__main__':
    yaml_file = "/Users/omernagar/Documents/Projects/wsc-interview/scripts/config_yamls/baseline_config.yaml"
    assert os.path.exists(yaml_file), f"yaml file {yaml_file} does not exist"
    with open(yaml_file, "r") as f:
        config_ = yaml.safe_load(f)

    train(config=config_)