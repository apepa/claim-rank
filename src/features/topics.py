from src.features.features import Feature
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from src.utils.config import get_config

CONFIG = get_config()


def get_bow(text, dict):
    return dict.doc2bow([t.lower() for t in word_tokenize(text)])


def topic_to_array(topics):
    return [t for _, t in topics]


class LDATopics(Feature):
    """Adds a vector of POS tag counts."""
    FEATS = ['lda']

    def __init__(self):
        self.lda = LdaModel.load(CONFIG['lda_model'])
        self.dict = Dictionary.load(CONFIG['lda_dict'])

    def transform(self, X):
        for sent in X:
            sent.features['lda'] = topic_to_array(self.lda[get_bow(sent.text, self.dict)])
        return X


class LDAVectorSim(Feature):
    """
    Adds a similarity between the LDA topic vector of the sentence and:
     - the LDA topic vector of the chunk;
     - the LDA topic vector of the next chunk;
     - the LDA topic vector of the previous chunk;
    """
    FEATS = ['lda_dist_prev', 'lda_dist_next', 'lda_dist']

    def __init__(self):
        self.lda = LdaModel.load(CONFIG['lda_model'])
        self.dict = Dictionary.load(CONFIG['lda_dict'])

    def transform(self, X):
        chunk_num = 0
        CHUNK_TEXTS = [""]
        MODERATORS = ['QUIJANO', 'COOPER', 'QUESTION', 'HOLT', 'WALLACE', 'RADDATZ']

        # split on chunks
        for i, sent in enumerate(X):
            # the moderator starts speaking - the start of the chunk
            if i > 0 \
                    and X[i - 1].speaker.strip().upper() not in MODERATORS \
                    and sent.speaker.strip().upper() in MODERATORS:
                chunk_num += 1
                # add the new chunk text to be filled
                CHUNK_TEXTS.append("")
            # keep the number of the chunk
            sent.features['mod_chunk'] = chunk_num
            CHUNK_TEXTS[chunk_num] = CHUNK_TEXTS[chunk_num] + " " + sent.text
        CHUNK_TEXTS.append("")
        #  add sim to chunk
        for sent in X:
            sent.features['lda_dist'] = self._get_sim(CHUNK_TEXTS[sent.features['mod_chunk']], sent.text)
            sent.features['lda_dist_prev'] = self._get_sim(CHUNK_TEXTS[sent.features['mod_chunk'] - 1], sent.text)
            sent.features['lda_dist_next'] = self._get_sim(CHUNK_TEXTS[sent.features['mod_chunk'] + 1], sent.text)
        return X

    def _get_sim(self, text1, text2):
        """
        :param text1:
        :param text2:
        :return: cosine similarity between the LDA vectors of the input texts.
        """
        text1_topic_vector = topic_to_array(self.lda[get_bow(text1, self.dict)])
        text2_topic_vector = topic_to_array(self.lda[get_bow(text2, self.dict)])
        sim = 1 - cosine(text1_topic_vector, text2_topic_vector)
        if np.isnan(sim):
            sim = 0
        return sim