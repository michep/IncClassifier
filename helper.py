"""
helper module
"""
from typing import Any, List
import pickle
import io
import pymorphy2
import stop_words
import nltk.tokenize
import pandas
import numpy


MORPH = pymorphy2.MorphAnalyzer()
TOKENIZER = nltk.tokenize.WordPunctTokenizer()
STOPWORDS = stop_words.get_stop_words(language="russian")


def load_csv(filename: str, encoding: str="utf8") -> pandas.DataFrame:
    """ load_csv """
    return pandas.read_csv(filename, encoding=encoding)


def save_pkl(obj: Any, filename: str, protocol: int=4) -> None:
    """ save_pkl """
    file = io.open(filename, "wb")
    pickle.dump(obj, file, protocol=protocol)
    file.close()


def load_pkl(filename: str) -> Any:
    """ load_pkl """
    file = io.open(filename, "rb")
    obj = pickle.load(file)
    file.close()
    return obj


def tokenize_str(line: str) -> List:
    """ tokenize_str """
    return [MORPH.parse(w)[0].normal_form for w in TOKENIZER.tokenize(line)
            if w not in STOPWORDS and w.isalpha()]


def normalize_str(line: str) -> str:
    """ normalize_str """
    return " ".join(tokenize_str(line))


def multiple_score(pred_mul: numpy.ndarray, test_mul: numpy.ndarray) -> int:
    """ multiple_score """
    correct = 0
    for rowi, row in enumerate(test_mul):
        correct_row = 0
        for eli, el in enumerate(row):
            if el == pred_mul[rowi][eli]:
                correct_row += 1
        if correct_row == len(row):
            correct += 1
    score = correct / len(test_mul)
    return score
