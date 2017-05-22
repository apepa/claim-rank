from nltk.tokenize import word_tokenize
from src.features.features import Feature


class Preprocess(Feature):
    def transform(self, X):
        for sent in X:
            sent.tokens = word_tokenize(sent.text)
        return X