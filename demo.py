import json
import os
import shutil

import streamlit as st

current_directory = os.path.dirname(os.path.abspath(__file__))
checkpoints_path = os.path.join(current_directory, 'app/src/models/checkpoints')
tmp_demo_folder = os.path.join(current_directory, 'app/assets/tmp_demo')

info_txt = os.path.join(tmp_demo_folder, 'info.txt')
results_json = os.path.join(tmp_demo_folder, 'results.json')

language = 'Русский'
st.set_page_config(layout="centered")


@st.cache_resource()
def get_quiz_generator():
    return QuizGenerator(checkpoints_path)


if not os.path.exists(info_txt):
    os.makedirs(tmp_demo_folder, exist_ok=True)
    page_num = 0
else:
    with open(info_txt) as f:
        lines = f.readlines()
        page_num = int(lines[0].strip())
        language, source_path = lines[1].strip(), lines[2].strip()

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');

        .big-font {
            font-size:70px !important;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
            margin-top: -20px;  /* Отрицательный отступ */
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="big-font">StudyMate</div>', unsafe_allow_html=True)

if language == 'Русский':
    st.sidebar.title("Выберите язык")
else:
    st.sidebar.title("Choose language")
language = st.sidebar.radio("Language:", ['Русский', 'English'], label_visibility='hidden')

if page_num == 0:

    st.write("StudyMate — инструмент, позволяющий по предоставленному образовательному материалу создать тест "
             "для проверки усвоенных знаний. StudyMate может работать с различными форматами входных данных: "
             "вы можете подать видео с лекцией или мастер-классом, запись образовательного подкаста или текст "
             "статьи или урока. Закрепляйте свои знания вместе с нашим инструментом!" if language == 'Русский' \
                 else "StudyMate is a tool that allows you to create a test based on the provided educational material "
                      "to check your knowledge. StudyMate can work with different input formats: you can submit a video"
                      " of a lecture or master class, a recording of an educational podcast, or the text of an article "
                      "or lesson. Reinforce your knowledge with our tool!")
    st.subheader('Try StudyMate:' if language == 'English' else 'Попробовать StudyMate:')

    uploaded_file = st.file_uploader("Upload a file" if language == 'English' else 'Загрузите файл',
                                     type=['txt', 'csv', 'mp4', 'mkv', 'wav', 'mp3'])

    if uploaded_file:
        with open(os.path.join(tmp_demo_folder, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        source_path = os.path.join(tmp_demo_folder, uploaded_file.name)
    else:
        source_path = None

    if source_path and not os.path.exists(source_path):
        error_text = "Input file doesn't exist" if language == 'English' else 'Такого файла не существует'
        st.error(error_text, icon="🚨")

    if st.button('Generate the test' if language == 'English' else 'Сгенерировать тест'):
        with open(info_txt, 'w') as f:
            f.writelines([f'{str(page_num + 1)}\n', f'{language}\n', f'{source_path}\n'])

        st.experimental_rerun()

elif page_num == 1:

    download_status = ['Generating test...', 'Initialization of models...',
                       'Test generation...'] if language == 'English' \
        else ['Генерируем тест...', 'Инициализация моделей...', 'Генерация теста...']

    with st.spinner(download_status[0]) as status:
        st.write(download_status[1])
        from app.src.quiz_generator import QuizGenerator

        quiz_generator = get_quiz_generator()
        st.write(download_status[2])
        results = quiz_generator.generate(source_path)

    with open(info_txt, 'w') as f:
        f.writelines([f'{str(page_num + 1)}\n', f'{language}\n', f'{source_path}\n'])

    with open(results_json, 'w') as f:
        json.dump(results, f)

    st.experimental_rerun()

elif page_num == 2:
    st.header('Ваш персональный тест' if language == 'Русский' else 'Your personal quiz')

    with open(results_json) as f:
        results = json.load(f)

    right_answers_count = 0

    for question in results:
        answer = st.radio(
            question['question'], list(question['choices'].keys())
        )
        if question['choices'][answer] == 1:
            right_answers_count += 1

    questions_count = len(results)

    if 'test_finished' not in st.session_state:
        st.session_state.test_finished = False

    if st.button('Завершить тест' if language == 'Русский' else 'Finish the test'):
        st.session_state.test_finished = True

        test_statistics = f'You answered {right_answers_count} questions out of {questions_count} correctly!' \
            if language == 'English' else f'Вы ответили правильно на {right_answers_count} вопросов из {questions_count}!'
        st.info(test_statistics, icon="ℹ️")

        accuracy = float(right_answers_count / questions_count)
        if accuracy <= 0.5:
            st.error('Тест не пройден!' if language == 'Русский' else 'Test failed!')
            st.snow()
        else:
            st.success('Тест успешно пройден!' if language == 'Русский' else 'Test passed!')
            st.balloons()

    if st.session_state.test_finished:
        if st.button('Попробовать снова' if language == 'Русский' else 'Try again'):
            st.session_state.test_finished = False
            shutil.rmtree(tmp_demo_folder)
            st.experimental_rerun()
