from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.utils.dicts import *
import nltk
from nltk.data import load
from nltk.tokenize import word_tokenize
from sklearn.pipeline import TransformerMixin


class Feature(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self


class Sentiment(Feature):
    """Adds sentiment of the text with NRC emotion lexicon"""
    def __init__(self):
        self.sent_lext = get_sentiment_lexicon()

    def transform(self, X, **transform_params):
        for sent in X:
            text = sent.text
            tokens = word_tokenize(text.lower())
            emotions_vector = [0 for _ in range(len(emotions))]
            for token in tokens:
                for i, emotion in enumerate(emotions):
                    emotions_vector[i] += self.sent_lext.get(token, {}).get(emotion, 0)
            sent.features['sentiment'] = emotions_vector

        return X


class TokenStat(Feature):
    """Adds token surface statistics"""
    COUNT_WORDS = ['you are', 'never', 'always', 'said', '?', 'biggest', 'worst', 'most', 'best', 'she'
                   'will', 'going to', 'would', 'stop', 'have to', 'not', 'of course', 'he']

    def transform(self, X, **transform_params):
        for sent in X:
            tokens = word_tokenize(sent.text.lower())
            sent.features['text_len'] = len(sent.text)
            sent.features['tokens_num'] = len(tokens)
            for word in self.COUNT_WORDS:
                sent.features[word] = sent.text.count(word)
        return X


class Speaker(Feature):
    """Adds the speaker of the sentence as a feature."""
    def __init__(self):
        self.speakers = {}

    def transform(self, X, **transform_params):
        for sent in X:
            if sent.speaker not in self.speakers:
                self.speakers[sent.speaker] = len(self.speakers)
            sent.features['speaker'] = [self.speakers[sent.speaker]]
        return X


class POS(Feature):
    """Adds a vector of POS tag counts."""
    def __init__(self):
        tagdict = load('help/tagsets/upenn_tagset.pickle')
        tags_keys = tagdict.keys()
        self.tags = {}
        for i, tag in enumerate(tags_keys):
            self.tags[tag] = i

    def transform(self, X, **transform_params):
        for sent in X:
            tokenized = word_tokenize(sent.text)
            tag_vector = [0 for _ in range(len(self.tags))]
            pos_tags = nltk.pos_tag(tokenized)
            for word, tag in pos_tags:
                tag_vector[self.tags[tag]] += 1
            sent.features['pos'] = tag_vector
        return X


class NER(Feature):
    """Adds NEs count"""
    def transform(self, X, **transform_params):
        for sent in X:
            tokens = word_tokenize(sent.text)
            parse_tree = nltk.ne_chunk(nltk.tag.pos_tag(tokens), binary=True)  # POS tagging before chunking!

            named_entities = []

            for subtree in parse_tree.subtrees():
                if subtree.label() == 'NE':
                    named_entities.append(subtree)
            sent.features['ner'] = len(named_entities)
        return X


class BagOfCounts(Feature):
    def __init__(self, training):
        self.vectorizer = CountVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          stop_words="english",
                                          max_features=5000)
        vocab = [s.text for s in training]
        self.vectorizer.fit_transform(vocab)

    def transform(self, X, **transform_params):
        for sent in X:
            sent.features['bag'] = self.vectorizer.transform([sent.text]).toarray().tolist()[0]
        return X


class BagOfTfIDF(Feature):
    def __init__(self, training):
        self.vectorizer = TfidfVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          ngram_range=(1, 1),
                                          stop_words="english",
                                          max_features=5000)
        vocab = [s.text for s in training]
        self.vectorizer.fit_transform(vocab)

    def transform(self, X, **transform_params):
        for sent in X:
            sent.features['bag_tfidf'] = self.vectorizer.transform([sent.text]).toarray().tolist()[0]
        return X

FEATURES = ['bag_tfidf', 'ner', 'pos', 'speaker', 'sentiment', 'text_len', 'tokens_num']
FEATURES += TokenStat.COUNT_WORDS


class ToMatrix(Feature):
    """Transforms the features dict to a matrix"""
    def transform(self, X, **transform_params):
        final_X = []

        for sent in X:
            sent_vector = []
            for feat in FEATURES:
                if isinstance(sent.features[feat], list):
                    sent_vector += sent.features[feat]
                else:
                    sent_vector.append(sent.features[feat])
            final_X.append(sent_vector)
        return final_X


def get_pipeline(sentences_train):
    return Pipeline([
        ('sentiment', Sentiment()),
        ('tokens', TokenStat()),
        ('pos', POS()),
        ('ner', NER()),
        ('speaker', Speaker()),
        ('tfidf', BagOfTfIDF(sentences_train)),
        #('counts', BagOfCounts(sentences_train)),
        ('transform', ToMatrix()),
        ('norm', Normalizer())
    ])
