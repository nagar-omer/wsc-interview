from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_curve
from wsc_interview.models.bert import get_bert_uncased_model
from wsc_interview import logger
from torch import nn, optim
import lightning as L
import numpy as np
import torch


OPTIMIZERS = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop
}


class ActionClassifier(nn.Module):
    def __init__(self, **kwargs):
        # unpack kwargs
        aggregate = kwargs.get("model", {}).get("aggregate_method", "mean")
        freeze_bert = kwargs.get("model", {}).get("freeze_bert", True)
        mlp_layers = kwargs.get("model", {}).get("mlp_layers", [768, 256, 1])

        logger.info(f"Creating action classifier with aggregation method: {aggregate}")
        super(ActionClassifier, self).__init__()

        if aggregate not in ['max', 'mean']:
            logger.error(f"Unknown aggregation method: {aggregate}")
        self._agg = aggregate

        self._bert_backbone = get_bert_uncased_model()
        # Freeze the BERT model parameters
        if freeze_bert:
            for param in self._bert_backbone.parameters():
                param.requires_grad = False

        # build linear layers
        mlp = []
        for i, (in_features, out_features) in enumerate(zip(mlp_layers[:-1], mlp_layers[1:])):
            mlp.append(nn.Linear(in_features, out_features))
            mlp.append(nn.ReLU() if i < len(mlp_layers) - 2 else nn.Sigmoid())
        self._mlp = nn.Sequential(*mlp)

    def forward(self, tokens, phrase_token_idx):
        # get bert output
        output = self._bert_backbone(tokens)
        # get token embeddings
        last_hidden_states = output.hidden_states[-1]

        # combine tokens
        if self._agg == 'max':
            tokens_embs = torch.stack([last_hidden_states[i][idx].max(axis=0).values for i, idx in enumerate(phrase_token_idx)])
        elif self._agg == 'mean':
            tokens_embs = torch.stack([last_hidden_states[i][idx].mean(axis=0) for i, idx in enumerate(phrase_token_idx)])
        else:
            raise ValueError(f"Unknown aggregation method: {self._agg}")

        # pass through mlp
        return self._mlp(tokens_embs)


class LitActionClassifier(L.LightningModule):
    def __init__(self, label_count=None, **kwargs):
        super().__init__()
        # store loss for plotting
        self._train_loss, self._val_loss = [], []

        # set loss function and class weights
        use_class_weights = kwargs.get("model", {}).get("weighted_loss", False)
        if not use_class_weights or label_count is None:
            self._class_weight = None
            self._criteria = nn.BCELoss()
        else:
            # [ neg weight, pos weight]
            pos_count = label_count[1]
            neg_count = label_count[0]
            self._class_weight = [
                (pos_count + neg_count) / (2.0 * neg_count),
                (pos_count + neg_count) / (2.0 * pos_count)
            ]
            self._criteria = nn.BCELoss(reduction='none')

        # Choose a specific version of CLIP, e.g., "openai/clip-vit-base-patch32"
        self._classifier = ActionClassifier(**kwargs)

        # Set optimizer
        self._optimizer = OPTIMIZERS[kwargs.get("optimizer", {}).get("name", "AdamW")]
        self._learning_rate = float(kwargs.get("optimizer", {}).get("learning_rate", 2e-5))
        self._weight_decay = float(kwargs.get("optimizer", {}).get("weight_decay", 0.01))

        # store predictions and labels for batch metrics calculation
        self._train_step_predictions, self._train_step_labels = [], []
        self._val_step_predictions, self._val_step_labels = [], []

        # set initial threshold
        self._threshold = 0.5

    @property
    def threshold(self):
        return self._threshold

    @property
    def classifier(self):
        return self._classifier

    def configure_optimizers(self):
        AdamW = optim.AdamW(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        kwargs = {'lr': self._learning_rate, 'weight_decay': self._weight_decay}
        optimizer = self._optimizer(self.parameters(), **kwargs)
        return optimizer

    def _auc(self, y_hat, y_true):
        # if all labels are the same, return 0.5
        if len(set(y_true.tolist())) == 1:
            return 0.5

        # calculate ROC-AUC
        y_hat = y_hat.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        return roc_auc_score(y_true, y_hat)

    def _set_threshold(self, y_hat, y_true):
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_hat)

        # Find the optimal threshold
        optimal_idx = np.argmin(np.sqrt(np.square(1 - tpr) + np.square(fpr)))
        optimal_threshold = thresholds[optimal_idx]
        self.log('optimal_threshold', optimal_threshold, on_epoch=True)
        self._threshold = optimal_threshold

    def _recall(self, y_hat, y_true):
        # calculate recall
        y_hat = y_hat.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        y_hat = (y_hat > self._threshold).astype(int)
        return recall_score(y_true, y_hat)

    def _precision(self, y_hat, y_true):
        # calculate precision
        y_hat = y_hat.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        y_hat = (y_hat > self._threshold).astype(int)
        return precision_score(y_true, y_hat)

    def training_step(self, batch, batch_idx):
        # Training step
        transcription, action_idx, phrase, y_true = batch

        y_hat = self._classifier(transcription, action_idx)
        if self._class_weight is None:
            loss = self._criteria(y_hat, y_true.unsqueeze(1).float())
        else:
            batch_weight = torch.ones_like(y_hat, dtype=torch.float) * self._class_weight[1]
            batch_weight[y_true == 0] = self._class_weight[0]
            loss = torch.mean(self._criteria(y_hat, y_true.unsqueeze(1).float()) * batch_weight)

        self.log('train_loss', loss, on_epoch=True)
        self._train_step_predictions.append(y_hat)
        self._train_step_labels.append(y_true)

        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        transcription, action_idx, phrase, y_true = batch

        y_hat = self._classifier(transcription, action_idx)
        if self._class_weight is None:
            loss = self._criteria(y_hat, y_true.unsqueeze(1).float())
        else:
            batch_weight = torch.ones_like(y_hat, dtype=torch.float) * self._class_weight[1]
            batch_weight[y_true == 0] = self._class_weight[0]
            loss = torch.mean(self._criteria(y_hat, y_true.unsqueeze(1).float()) * batch_weight)

        self.log('val_loss', loss, on_epoch=True)  # Log loss for the entire validation set
        self._val_step_predictions.append(y_hat)
        self._val_step_labels.append(y_true)

        return loss

    def on_train_epoch_end(self):
        # unpack predictions and labels
        pred, labels = torch.cat(self._train_step_predictions), torch.cat(self._train_step_labels)

        # calculate metrics
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        auc = self._auc(pred, labels)
        recall = self._recall(pred, labels)
        precision = self._precision(pred, labels)
        self._train_step_predictions, self._train_step_labels = [], []

        print(f'\n\nEpoch [{self.current_epoch + 1}/{self.trainer.max_epochs}] - '
              f'Training Loss: {train_loss:.4f}', f'AUC: {auc:.4f}', f'Recall: {recall:.4f}',
              f'Precision: {precision:.4f}')
        self._train_loss.append(train_loss)

    def on_validation_epoch_end(self):
        # unpack predictions and labels
        pred, labels = torch.cat(self._val_step_predictions), torch.cat(self._val_step_labels)

        # set threshold
        self._set_threshold(pred, labels)

        # get metrics
        val_loss = self.trainer.callback_metrics['val_loss'].item()
        auc = self._auc(pred, labels)
        recall = self._recall(pred, labels)
        precision = self._precision(pred, labels)
        self._val_step_predictions, self._val_step_labels = [], []

        print(f'\n\nEpoch [{self.current_epoch + 1}/{self.trainer.max_epochs}] - '
              f'Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}', f'Recall: {recall:.4f}',
                f'Precision: {precision:.4f}')
        self._val_loss.append(val_loss)
