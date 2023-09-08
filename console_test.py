# coding=utf-8

from app.src.quiz_generator import QuizGenerator


def main():
    checkpoints_path = 'app/src/models/checkpoints'
    source_path = 'app/assets/stepik_text_example.txt'

    quiz_generator = QuizGenerator(checkpoints_path)
    results = quiz_generator.generate(source_path)

    for res in results:
        chunk_context = res['context']
        question = res['question']
        choices = res['choices']
        print(f'Отрывок: {chunk_context}')
        print()
        print(f'Вопрос: {question}')
        print(f'Варианты ответа:')
        for i, (k, v) in enumerate(choices.items()):
            print(f'\t{i + 1}. {k}{" <-- Правильный" if v == 1 else ""}')
        print('----------------------------------------------------------')


if __name__ == '__main__':
    main()
