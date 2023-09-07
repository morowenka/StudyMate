import re
import string
import warnings
from collections import OrderedDict
from typing import List

import nltk
import pytorch_lightning as pl
import torch
from sense2vec import Sense2Vec
from transformers import T5TokenizerFast as T5Tokenizer

from app.src.models.base_models.qg_model import QGModel

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings('ignore')

DistrG_MODEL_NAME = 't5-small'
DistrG_SOURCE_MAX_TOKEN_LEN = 512
DistrG_TARGET_MAX_TOKEN_LEN = 64

DEVICE = torch.device('cuda')
if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')

COUNT_OF_DISTRACTORS = 3
QnAG_SEP_TOKEN = '<sep>'

pl.seed_everything(42)


def _replace_all_extra_id(text: str) -> str:
    pattern = r'<extra_id_\d+>'
    return re.sub(pattern, QnAG_SEP_TOKEN, text)


def _is_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


class DistractorGenerator:

    def __init__(self, checkpoint_path):
        self.tokenizer = T5Tokenizer.from_pretrained(DistrG_MODEL_NAME)
        self.tokenizer.add_tokens(QnAG_SEP_TOKEN)
        self.tokenizer_len = len(self.tokenizer)

        self.dg_model = QGModel.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
        self.dg_model.freeze()
        self.dg_model.eval()

    def generate(self, generate_count: int, correct: str, question: str, context: str) -> List[str]:
        if _is_numeric(correct):
            return self._generate_numeric_distractors(correct, generate_count)

        generate_triples_count = int(generate_count / 3) + 1
        model_output = self._model_predict(generate_triples_count, correct, question, context)

        cleaned_result = model_output.replace('<pad>', '').replace('</s>', QnAG_SEP_TOKEN)
        cleaned_result = _replace_all_extra_id(cleaned_result)
        distractors = cleaned_result.split(QnAG_SEP_TOKEN)[:-1]
        distractors = [x.translate(str.maketrans('', '', string.punctuation)).strip() for x in distractors]

        return distractors

    def _generate_numeric_distractors(self, correct: str, count: int) -> List[str]:
        number = float(correct)
        distractors = []

        for i in range(1, count + 1):
            if number > 0:
                distractors.append(str(number + i))
                distractors.append(str(number - i))
            else:
                distractors.append(str(number - i))
                distractors.append(str(number + i))

        if "." not in correct:
            distractors = [str(int(float(d))) for d in distractors]
        return distractors[:count]

    def _model_predict(self, generate_count: int, correct: str, question: str, context: str) -> str:
        source_encoding = self.tokenizer(
            f'{correct} {QnAG_SEP_TOKEN} {question} {QnAG_SEP_TOKEN} {context}',
            max_length=DistrG_SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        generated_ids = self.dg_model.model.generate(
            input_ids=source_encoding['input_ids'],
            attention_mask=source_encoding['attention_mask'],
            num_beams=2 * generate_count,
            num_return_sequences=generate_count,
            max_length=DistrG_TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        return ''.join([
            self.tokenizer.decode(gid, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for gid in generated_ids
        ])


class Sense2VecDistractorGeneration:
    def __init__(self, checkpoint_path):
        self.s2v = Sense2Vec().from_disk(checkpoint_path)

    def generate(self, answer: str, desired_count: int) -> List[str]:
        answer_normalized = answer.lower().replace(" ", "_")
        sense = self.s2v.get_best_sense(answer_normalized)

        if not sense:
            return []

        distractors = self._get_distractors_from_sense(sense, answer_normalized, desired_count)
        return list(OrderedDict.fromkeys(distractors))

    def _get_distractors_from_sense(self, sense, answer, count):
        distractors = []
        most_similar = self.s2v.most_similar(sense, n=count)

        for phrase in most_similar:
            normalized_phrase = phrase[0].split("|")[0].replace("_", " ").lower()

            if normalized_phrase != answer:
                distractors.append(normalized_phrase.capitalize())

        return distractors
