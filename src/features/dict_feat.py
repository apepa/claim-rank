from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import Normalizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.data import load
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, tag, pos_tag

from src.data.debates import DEBATES, read_debates
from src.utils.dicts import *
from src.features.features import Feature, ToMatrix

class Sentiment_NRC(Feature):
    """Adds sentiment of the text with NRC emotion lexicon"""
    def __init__(self):
        self.sent_lext = get_sentiment_lexicon()

    def transform(self, X):
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
    COUNT_WORDS = ['you are', 'never', 'always', 'said', '?', 'biggest', 'worst', 'most', 'best', 'she',
                   'will', 'going to', 'would', 'stop', 'have to', 'not', 'of course', 'he']

    def transform(self, X):
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

    def transform(self, X):
        for sent in X:
            if sent.speaker not in self.speakers:
                self.speakers[sent.speaker.strip()] = len(self.speakers)
            sent.features['speaker'] = self.speakers[sent.speaker.strip()]
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

    def transform(self, X):
        for sent in X:
            sent.features['bag'] = self.vectorizer.transform([sent.text]).toarray().tolist()[0]
        return X


class BagOfTfIDFNGR(Feature):
    def __init__(self, training):
        self.vectorizer = TfidfVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          ngram_range=(1, 1),
                                          stop_words="english",
                                          max_features=5000)
        vocab = [s.text for s in training]
        self.vectorizer.fit_transform(vocab)

    def transform(self, X):
        for sent in X:
            sent.features['bag_tfidf'] = self.vectorizer.transform([sent.text]).toarray().tolist()[0]
        return X

FEATURES = ['bag_tfidf', 'speaker', 'text_len', 'tokens_num']
FEATURES += TokenStat.COUNT_WORDS

SYSTEM = ['(laugh', '(crosstalk', '(applause', '(laughter']



class SyntacticParse(Feature):
    def __init__(self):
        self.syntactic_parses = {}

        for debate in DEBATES:
            parsed = open("../../data/parses/"+CONFIG[debate.name]+"_parsed.txt")
            sentences = read_debates(debate)
            for sentence in sentences:
                parse = [float(x) for x in parsed.readline().strip().split()[1:]]

                self.syntactic_parses[sentence.debate.name+sentence.id] = parse

    def transform(self, X):
        for sent in X:
            sent.features['syntctic_parse'] = self.syntactic_parses[sent.debate.name+sent.id]
            print(len(self.syntactic_parses[sent.debate.name+sent.id]))
        return X

class Negatives(Feature):
    def __init__(self):
        self.contras = get_contra_vocab()
        self.negatives = get_negative_vocab()

    def transform(self, X):
        for i, sent in enumerate(X):
            sent.features['negs'] = self.count_neg(sent.text)
            sent.features['contras'] = self.count_contras(sent.text)
            sent.features['negs_next'] = 0
            sent.features['contras_next'] = 0
            if i <= len(X)-1 and sent.debate == X[i+1].debate:
                sent.features['negs_next'] = self.count_neg(X[i+1].text)
                sent.features['contras_next'] = self.count_contras(X[i + 1].text)
        return X

    def count_neg(self, text):
        tokenized = word_tokenize(text.lower())
        result = 0
        for neg in self.negatives:
            result += tokenized.count(neg)
        return result

    def count_contras(self, text):
        tokenized = word_tokenize(text.lower())
        result = 0
        for contra in self.contras:
            result += tokenized.count(contra)
        return result


class ChunkLen(Feature):

    def transform(self, X):
        CHUNKS = {}
        curr_chunk_id = 0
        curr_speaker = X[0].speaker
        curr_chunk_len = 0
        for i, sent in enumerate(X):
            if curr_speaker != sent.speaker or i == len(X)-1:
                CHUNKS[curr_chunk_id] = curr_chunk_len
                curr_speaker = sent.speaker
                curr_chunk_id += 1
                curr_chunk_len = 1
            else:
                curr_chunk_len += 1
            sent.features['chunk_id'] = curr_chunk_id
        CHUNKS[-1] = 0
        CHUNKS[curr_chunk_id] = curr_chunk_len
        CHUNKS[curr_chunk_id+1] = 0

        k = 0
        curr_speaker = X[0].speaker
        for i, sent in enumerate(X):
            if curr_speaker != sent.speaker:
                curr_speaker = sent.speaker
                X[i-1].features['last'] = 1
                k = 1
            else:
                k += 1
            sent.features['last'] = 0
            sent.features['first'] = 1 if k == 1 else 0
            sent.features['chunk_size'] = CHUNKS[sent.features['chunk_id']]
            sent.features['prev_chunk_size'] = CHUNKS[sent.features['chunk_id'] - 1]
            sent.features['next_chunk_size'] = CHUNKS[sent.features['chunk_id'] + 1]
            sent.features['number_in_chunk'] = k

            if i>1 and sent.debate != X[i-1].debate:
                sent.features['prev_chunk_size'] = 0
                X[i-1].features['next_chunk_size'] = 0

        return X

class System(Feature):

    def transform(self, X):
        for i, sent in enumerate(X):
            for feat in SYSTEM:
                if i+1<len(X) and feat in X[i+1].text.lower():
                    sent.features[feat] = 1
                else:
                    sent.features[feat] = 0
        return X


class NegationNextChunk(Negatives):
    def transform(self, X):
        for i, sent in enumerate(X):
            j = i+1
            while X[j].speaker == X[i].speaker:
                j += 1

            k = j
            count_next_neg = 0
            count_next_contra = 0
            if X[j].debate == X[i].debate:
                while X[k].speaker == X[j].speaker:
                    count_next_neg += self.count_neg(X[k].text)
                    count_next_contra += self.count_contras(X[k].text)
                    k += 1
                sent.features['negs_next_chunk'] = count_next_neg
                sent.features['contras_next_chunk'] = count_next_contra
        return X

def get_pipeline(sentences_train):
    return Pipeline([
        # ('sentiment', Sentiment_NRC()),
        ('tokens', TokenStat()),
        # ('pos', POS()),
        # ('ner', NER_NLTK()),
        ('speaker', Speaker()),
        # ('tfidf', BagOfTfIDF(sentences_train)),
        # ('counts', BagOfCounts(sentences_train)),
        ('transform', ToMatrix()),
        ('norm', Normalizer())
    ])




ALL_FEATS = ['negs', 'contras', 'chunk_id', 'last', 'first', 'chunk_size','prev_chunk_size', 'next_chunk_size','number_in_chunk',
'negation_next_chunk','negation_next', 'neg_count','negation' , 'sent','len','bag_tfidf','pos','ner',
             'syntctic_parse']
ALL_FEATS += TokenStat.COUNT_WORDS
ALL_FEATS += SYSTEM

f = ['chunk_size']
def get_pipeline_new(sentences_train):
    return Pipeline([
        # ('sentiment', Sentiment()),
        # ('speaker', Speaker()),
        # ('system', System()),
        # ('sent_len', SentenceLength()),
        # ('ner', NER()),
        # ('pos', POS()),
        # ('syn', SyntacticParse()),
        # ('tfidf', BagOfTfIDFNGR(sentences_train)),
        # ('chunks', ChunkLen()),
        # ('negatives', Negatives()),

        # ('neg_chunk', NegationNextChunk()),

        # ('tokens', TokenStat()),

        ('transform', ToMatrix(features=f)),
        ('norm', MinMaxScaler())
    ])