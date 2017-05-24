from src.features.features import Feature
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize


class BagOfTfIDF(Feature):
    """Adds Bag of TF-IDF scores of words.
    This is used in ClaimBuster approach."""
    FEATS = ['bag_tfidf']

    def __init__(self, training):
        self.vectorizer = TfidfVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          ngram_range=(1, 1),
                                          min_df=3)
        vocab = [s.text for s in training]
        self.vectorizer.fit_transform(vocab)

    def transform(self, X):
        for sent in X:
            sent.features['bag_tfidf'] = self.vectorizer.transform([sent.text]).toarray().tolist()[0]
        return X


class BagOfTfIDFN(Feature):
    """Adds Bag of TF-IDF scores of 1/2/3-grams."""
    FEATS = ['bag_tfidf_n']

    def __init__(self, training):
        self.vectorizer = TfidfVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          ngram_range=(1, 4),
                                          min_df=10,
                                          max_df=0.4,
                                          stop_words="english",
                                          max_features=2000)
        vocab = [s.text for s in training]
        self.vectorizer.fit_transform(vocab)

    def transform(self, X):
        for sent in X:
            sent.features['bag_tfidf_n'] = self.vectorizer.transform([sent.text]).toarray().tolist()[0]
        return X


class BagOfCounts(Feature):
    """Adds Bag of Counts of words."""
    FEATS = ['bag']

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


class TokenStat(Feature):
    """Adds specific token counts."""
    FEATS = ['America', 'Reagan', 'Mexico', 'tax', 'i ', 'said', 'have to', 'you ']

    def transform(self, X):
        for sent in X:
            for word in self.FEATS:
                sent.features[word] = sent.text.count(word)
        return X


class SentenceLength(Feature):
    FEATS = ['tokens_num', 'text_len']

    def transform(self, X):
        for sent in X:
            # this counts the punctuation, too
            # TODO add separate count for puntuation
            sent.features['tokens_num'] = len(word_tokenize(sent.text))
            sent.features['text_len'] = len(sent.text)
        return X


class ChunkLen(Feature):
    """Adds length of the current, next and previous chunks.
    Chunk is a sequence of sentences said by one person."""
    FEATS = ['last', 'first', 'chunk_size', 'prev_chunk_size', 'next_chunk_size', 'number_in_chunk']

    def transform(self, X):
        CHUNKS = {}
        curr_chunk_id = 0
        curr_speaker = X[0].speaker
        curr_chunk_len = 0
        for i, sent in enumerate(X):
            if curr_speaker != sent.speaker or i == len(X) - 1:
                CHUNKS[curr_chunk_id] = curr_chunk_len
                curr_speaker = sent.speaker
                curr_chunk_id += 1
                curr_chunk_len = 1
            else:
                curr_chunk_len += 1
            sent.features['chunk_id'] = curr_chunk_id
        CHUNKS[-1] = 0
        CHUNKS[curr_chunk_id] = curr_chunk_len
        CHUNKS[curr_chunk_id + 1] = 0

        num_in_chunk = 0
        curr_speaker = X[0].speaker
        for i, sent in enumerate(X):
            if curr_speaker != sent.speaker:
                curr_speaker = sent.speaker
                X[i - 1].features['last'] = 1
                num_in_chunk = 1
            else:
                num_in_chunk += 1
            sent.features['last'] = 0
            sent.features['first'] = 1 if num_in_chunk == 1 else 0
            sent.features['chunk_size'] = CHUNKS[sent.features['chunk_id']]
            sent.features['prev_chunk_size'] = CHUNKS[sent.features['chunk_id'] - 1]
            sent.features['next_chunk_size'] = CHUNKS[sent.features['chunk_id'] + 1]
            sent.features['number_in_chunk'] = num_in_chunk

            if i > 1 and sent.debate != X[i - 1].debate:
                sent.features['prev_chunk_size'] = 0
                X[i - 1].features['next_chunk_size'] = 0

        return X
