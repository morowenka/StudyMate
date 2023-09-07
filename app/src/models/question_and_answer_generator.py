import warnings
from typing import Tuple

import nltk
import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast as T5Tokenizer

from app.src.models.base_models.qg_model import QGModel

nltk.download('stopwords')
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda')
if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')

QnAG_MODEL_NAME = 't5-small'
QnAG_SOURCE_MAX_TOKEN_LEN = 300
QnAG_TARGET_MAX_TOKEN_LEN = 80
QnAG_SEP_TOKEN = '<sep>'
QnAG_ANSWER_MASK = '[MASK]'

pl.seed_everything(42)


class QuestionGenerator:

    def __init__(self, checkpoint_path):
        self.tokenizer = T5Tokenizer.from_pretrained(QnAG_MODEL_NAME)
        self.tokenizer.add_tokens(QnAG_SEP_TOKEN)
        self.qg_model = QGModel.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        self.qg_model.freeze()
        self.qg_model.eval()

    def generate(self, answer: str, context: str) -> str:
        _, question = self._model_predict(answer, context).split(QnAG_SEP_TOKEN)
        return question

    def generate_qna(self, context: str) -> Tuple[str, str]:
        model_output = self._model_predict(QnAG_ANSWER_MASK, context)
        return tuple(model_output.split(f'{QnAG_SEP_TOKEN} ')) if QnAG_SEP_TOKEN in model_output else ('', model_output)

    def _model_predict(self, answer: str, context: str) -> str:
        source_encoding = self.tokenizer(
            f'{answer} {QnAG_SEP_TOKEN} {context}',
            max_length=QnAG_SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        generated_ids = self.qg_model.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            num_beams=16,
            max_length=QnAG_TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        return ''.join([
            self.tokenizer.decode(gid, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gid in generated_ids
        ])
