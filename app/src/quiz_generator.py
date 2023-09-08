import os
import warnings
import zipfile

import gdown
import numpy as np
from googletrans import Translator, LANGUAGES
from punctuators.models import PunctCapSegModelONNX

from app.src.models.distractors_generators import DistractorGenerator, Sense2VecDistractorGeneration
from app.src.models.question_and_answer_generator import QuestionGenerator
from app.src.processing.distractor_processing import DistractorProcessor
from app.src.processing.text_extractor import extract_text
from app.src.processing.text_processing import TextCleaner, remove_trailing_symbols, split_text_into_chunks
from app.src.models.translator import TranslatorModel

warnings.filterwarnings('ignore')


class QuizGenerator:
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path
        if not [f for f in os.listdir(self.checkpoints_path) if f != ".gitkeep"]:
            print('downloading models from google drive...')
            self.download_from_google_drive()

        print(os.listdir(self.checkpoints_path))

        self.qg_checkpoint_path = os.path.join(self.checkpoints_path, 'multitask-qg-ag.ckpt')
        self.dist_checkpoint_path = os.path.join(self.checkpoints_path, 'race-distractors.ckpt')
        self.s2v_checkpoint_path = os.path.join(self.checkpoints_path, 's2v_old')

        self.question_generator = QuestionGenerator(self.qg_checkpoint_path)
        self.distractor_generator = DistractorGenerator(self.dist_checkpoint_path)
        self.sense2vec_distractor_generator = Sense2VecDistractorGeneration(self.s2v_checkpoint_path)
        self.grammar_model = PunctCapSegModelONNX.from_pretrained(
            '1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase')

        self.google_translator = Translator()
        self.translator = TranslatorModel()
        self.processor = DistractorProcessor()

        self.COUNT_OF_DISTRACTORS = 3

    def process_chunk(self, chunk):
        original_chunk = chunk
        lang_detected_original = self.google_translator.detect(chunk)
        if LANGUAGES[lang_detected_original.lang] == 'russian':
            chunk = self.translator.translate(chunk, src='ru', dest='en')

        cleaner = TextCleaner(chunk)
        cleaned_text = cleaner.clean()

        answer, question = self.question_generator.generate_qna(cleaned_text)

        t5_distractors = self.distractor_generator.generate(self.COUNT_OF_DISTRACTORS * 3, answer, question,
                                                            cleaned_text)

        unique_distractors = self.processor.remove_duplicates(t5_distractors)
        filtered_distractors = self.processor.remove_distractors_duplicate_with_correct_answer(answer,
                                                                                               unique_distractors)

        required_distractors = self.COUNT_OF_DISTRACTORS - len(filtered_distractors)
        if required_distractors > 0:
            s2v_distractors = self.sense2vec_distractor_generator.generate(answer, required_distractors * 2)
            distractors = filtered_distractors + s2v_distractors
            unique_distractors = self.processor.remove_duplicates(distractors)
            filtered_distractors = self.processor.remove_distractors_duplicate_with_correct_answer(answer,
                                                                                                   unique_distractors)

        if len(filtered_distractors) < self.COUNT_OF_DISTRACTORS:
            print("Warning: Unable to generate sufficient unique distractors!")

        distractors = filtered_distractors[:self.COUNT_OF_DISTRACTORS]

        distractors = [dist[0] for dist in self.grammar_model.infer(texts=distractors, apply_sbd=True)]
        answer = self.grammar_model.infer(texts=[answer], apply_sbd=True)[0][0]

        if LANGUAGES[lang_detected_original.lang] == 'russian':
            question = self.translator.translate(question, src='en', dest='ru')

            distractors = [self.translator.translate(dist, src='en', dest='ru') for dist in distractors]
            answer = self.translator.translate(answer, src='en', dest='ru')

        distractors = [remove_trailing_symbols(dist) for dist in distractors]
        answer = remove_trailing_symbols(answer)

        choices = {answer: 1}
        for dist in distractors:
            choices[dist] = 0
        items = list(choices.items())
        np.random.shuffle(items)
        shuffled_choices = dict(items)

        return {
            "context": original_chunk,
            "question": question,
            "choices": shuffled_choices
        }

    def download_from_google_drive(self):
        url = "https://drive.google.com/uc?id=1Q5AvkSJDVABcK-9MydvU7HUzlUhYDYEz"
        output_path = os.path.join(self.checkpoints_path, 'checkpoints.zip')
        gdown.download(url, output_path, quiet=False)
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(self.checkpoints_path)
        os.remove(output_path)

    def generate(self, source_path, count_of_distractors=3):
        self.COUNT_OF_DISTRACTORS = count_of_distractors
        context = extract_text(source_path)
        chunks = split_text_into_chunks(context)
        results = []
        for chunk in chunks:
            result = self.process_chunk(chunk)
            results.append(result)
        return results
