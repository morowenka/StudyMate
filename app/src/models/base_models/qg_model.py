import warnings

import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration

warnings.filterwarnings('ignore')

QnAG_MODEL_NAME = 't5-small'
QnAG_LEARNING_RATE = 0.0001
QnAG_TOKENIZER_LEN = 32101

pl.seed_everything(42)


class QGModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(QnAG_MODEL_NAME, return_dict=True)
        self.model.resize_token_embeddings(QnAG_TOKENIZER_LEN)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=QnAG_LEARNING_RATE)
