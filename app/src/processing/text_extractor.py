import io
import os
import threading

import speech_recognition as sr
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence


def extract_text(file_path):
    def _choose_your_method(path):
        name, extension = os.path.splitext(path)
        file_name = name[0:].lower()
        file_extension = extension[1:].lower()
        if file_extension in ["txt", "csv"]:
            if file_extension in ["txt", "csv"]:
                with open(path, "r", encoding="utf-8") as file:
                    return file.read()
        elif file_extension in ["mp4", "mkv"]:
            audio_name = file_name + ".wav"
            _extract_audio_from_video(path, audio_name)
            return _get_audio_transcription(audio_name, "wav")
        elif file_extension in ["wav", "mp3"]:
            return _get_audio_transcription(path, file_extension)
        else:
            return "Unknown"

    def _extract_audio_from_video(video_path, output_audio_path):
        audioclip = AudioFileClip(video_path)
        audioclip.write_audiofile(output_audio_path)

    def _recognize_and_transcribe(audio_chunk, text_list, i, file_extension):
        try:
            r = sr.Recognizer()
            with io.BytesIO() as f:
                audio_chunk.export(f, format=file_extension)
                f.seek(0)
                audio_file = sr.AudioFile(f)
                with audio_file as source:
                    audio_data = r.record(source)
                text = r.recognize_google(audio_data, language='ru-RU')
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            text_list.append((i, text))

    def _get_audio_transcription(path, file_extension):
        sound = AudioSegment.from_file(path)
        chunks = split_on_silence(sound,
                                  min_silence_len=700,
                                  silence_thresh=sound.dBFS - 14,
                                  keep_silence=300,
                                  )
        threads = []
        whole_text = []
        for i, audio_chunk in enumerate(chunks, start=1):
            thread = threading.Thread(target=_recognize_and_transcribe,
                                      args=(audio_chunk, whole_text, i, file_extension))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        whole_text.sort(key=lambda x: x[0])
        text_list = [text for _, text in whole_text]
        return " ".join(text_list)

    return _choose_your_method(file_path)
