from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Literal


class TranslatorModel:
    def __init__(self, model_name_prefix: str = 'Helsinki-NLP/opus-mt-'):
        ru_en_model_name = f'{model_name_prefix}ru-en'
        self.ru_en_tokenizer = AutoTokenizer.from_pretrained(ru_en_model_name)
        self.ru_en_model = AutoModelForSeq2SeqLM.from_pretrained(ru_en_model_name)

        en_ru_model_name = f'{model_name_prefix}en-ru'
        self.en_ru_tokenizer = AutoTokenizer.from_pretrained(en_ru_model_name)
        self.en_ru_model = AutoModelForSeq2SeqLM.from_pretrained(en_ru_model_name)

    def translate(
        self, text: str, src: Literal['en', 'ru'],
        dest: Literal['en', 'ru'], max_new_tokens: int = 1000,
    ):
        tokenizer = self.ru_en_tokenizer if src == 'ru' else self.en_ru_tokenizer
        model = self.ru_en_model if src == 'ru' else self.en_ru_model

        inputs = tokenizer(text, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return out_text
