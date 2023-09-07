import re

from nltk.tokenize import sent_tokenize, word_tokenize


class TextCleaner:
    def __init__(self, text: str):
        self.text = text

    def remove_brackets_content(self):
        self.text = re.sub(r'\((.*?)\)', '', self.text)
        self.text = re.sub(r'\[(.*?)\]', '', self.text)
        return self

    def remove_multiple_spaces(self):
        self.text = re.sub(' +', ' ', self.text)
        return self

    def replace_weird_hyphen(self):
        self.text = self.text.replace('–', '-')
        return self

    def clean(self):
        return self.remove_brackets_content().remove_multiple_spaces().replace_weird_hyphen().text


def split_text_into_chunks(context, max_n_tokens=300):
    sents = sent_tokenize(context)
    chunks = []
    sent_idx = 0
    while sent_idx < len(sents):
        chunk_words = 0
        chunks.append("")
        while chunk_words <= max_n_tokens and sent_idx < len(sents):
            curr_sent = sents[sent_idx]
            curr_sent_words = word_tokenize(curr_sent)
            if len(curr_sent_words) + chunk_words <= max_n_tokens:
                sent_idx += 1
                chunks[-1] += curr_sent
            chunk_words += len(curr_sent_words)
    return chunks


def remove_trailing_symbols(text):
    return text.rstrip('.?!, ')
