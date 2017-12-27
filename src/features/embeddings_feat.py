import gensim
from src.features.features import Feature
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from src.utils.config import get_config
import numpy as np

CONFIG = get_config()
w2v = gensim.models.KeyedVectors.load_word2vec_format(CONFIG['w2v'], binary=True)


def get_sent_vetor(text):
    """
    Get text's word2vec embedding vector, which is the mean of the w2v vector of the words in it.
    :return: 300-dim array, representing the document in the w2v cector space.
    """
    tokenized = text if text == "" else word_tokenize(text)
    all_word_vectors = [np.array(w2v[w]) for w in tokenized if w in w2v]
    if len(all_word_vectors) == 0:
        result = np.zeros(300)
    else:
        result = np.mean(all_word_vectors, axis=0)
    return [float(i) for i in result]


def get_sim(text1, text2):
    """
    Computes the cosine similarity between the w2v vectors of two documents.
    :return: float in [0,1]
    """
    a = get_sent_vetor(text1)
    b = get_sent_vetor(text2)
    sim = 1 - cosine(a, b)
    if np.isnan(sim):
        sim = 0
    return sim


class W2VVectors(Feature):
    """
    Adds as a feature the w2v sentence vector.
    """
    FEATS = ['w2v_vector']

    def transform(self, X):
        for sent in X:
            sent.features['w2v_vector'] = get_sent_vetor(sent.text)
        return X


class W2VVectorSim(Feature):
    """
    Adds as a feture the w2v similarity between the sentence and:
        - the current chunk
        - the next chunk
        - the previous chunk

    The chunks end when a moderator start talking, introducing new topic in the debate.
    """
    FEATS = ['w2v_dist', 'w2v_dist_prev', 'w2v_dist_next']
    MODERATORS = ['QUIJANO', 'COOPER', 'QUESTION', 'HOLT', 'WALLACE', 'RADDATZ']

    def transform(self, X):
        chunk_number = 0
        chunk_texts = [""]

        # find the chunks
        for i, sent in enumerate(X):
            if i > 0 \
                    and X[i - 1].speaker.strip().upper() not in self.MODERATORS \
                    and sent.speaker.strip().upper() in self.MODERATORS:
                chunk_number += 1
                chunk_texts.append("")
            sent.features['mod_chunk'] = chunk_number
            chunk_texts[chunk_number] = chunk_texts[chunk_number] + " " + sent.text
        chunk_texts.append("")

        #  add sim to chunk
        for sent in X:
            sent.features['w2v_dist'] = get_sim(chunk_texts[sent.features['mod_chunk']], sent.text)
            sent.features['w2v_dist_prev'] = get_sim(chunk_texts[sent.features['mod_chunk']-1], sent.text)
            sent.features['w2v_dist_next'] = get_sim(chunk_texts[sent.features['mod_chunk']+1], sent.text)
        return X