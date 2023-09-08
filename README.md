# StudyMate

## English version

### Intro

StudyMate is a tool that allows you to create a test based on the provided educational material to check your knowledge. StudyMate can work with different input formats: you can submit a video of a lecture or master class, a recording of an educational podcast, or the text of an article or lesson. Reinforce your knowledge with our tool!

### Virtual environment and running code

To run the application you must first:

1. build either a virtual environment by installing the packages specified in the file ```requirements.txt```, or a docker image (file ```Dockerfile```)
2. put the weights of the pre-trained models in the folder ```./app/src/models/checkpoints``` (they can be downloaded here: https://drive.google.com/drive/folders/174MoVolzQQJE5hmgIaF4wBTTAFUwJQIZ?usp=drive_link)

After that, launch using one of the commands:

1. ```streamlit run demo.py```: running an interactive stand on streamlit with the ability to upload an input file, pass the generated test and summarize the results
2. ```python3 console_test.py``` (first change the value of the ```source_path``` variable to the path to the input file): as a result of running, the generated test will be output to the console in text form, indicating the correct answers

## Версия на русском

### Введение

StudyMate — инструмент, позволяющий по предоставленному образовательному материалу создать тест для проверки усвоенных знаний. StudyMate может работать с различными форматами входных данных: вы можете подать видео с лекцией или мастер-классом, запись образовательного подкаста или текст статьи или урока. Закрепляйте свои знания вместе с нашим инструментом!

### Виртуальное окружение и запуск кода

Для запуска приложения необходимо предварительно:

1. собрать либо виртуальное окружение, установив пакеты, указанные в файле ```requirements.txt```, либо docker-образ (файл ```Dockerfile```)
2. положить веса предобученных моделей в папку ```./app/src/models/checkpoints``` (их можно скачать тут: https://drive.google.com/drive/folders/174MoVolzQQJE5hmgIaF4wBTTAFUwJQIZ?usp=drive_link)

После этого осуществить запуск с помощью одной из команд:

1. ```streamlit run demo.py```: запуск интерактивного стенда на streamlit с возможностью подгрузки входного файла, прохождения сгенерированного теста и подведением итогов
2. ```python3 console_test.py``` (предварительно поменять значение переменной ```source_path``` на путь до входного файла): в результате запуска в консоль будут выведен сгенерированный тест в текстовом виде, с указанием правильных ответов