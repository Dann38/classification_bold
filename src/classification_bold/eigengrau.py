import numpy


def coefficient(word: numpy.ndarray) -> float:
    h0 = word.shape[0]
    word[word > 22] = 255
    word = 255-word
    rez = word.mean()/h0
    return rez
