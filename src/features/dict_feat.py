from nltk.tokenize import word_tokenize
from os.path import join
from src.utils.dicts import *
from src.features.features import Feature
from nltk import pos_tag
import codecs
# extracts sentence-level features / Sentiment
class Sentiment_NRC(Feature):
    """Adds sentiment of the text with NRC emotion lexicon"""
    print("Calculating ... Sentiment_NRC | features['sent_nrc']")
    FEATS = ['sent_nrc']

    def __init__(self):
        self.sent_lext = get_sentiment_lexicon()

    def transform(self, X):
        for sent in X:
            emotions_vector = [0 for _ in range(len(emotions))]
            for token in sent.tokens:
                for i, emotion in enumerate(emotions):
                    emotions_vector[i] += self.sent_lext.get(token, {}).get(emotion, 0)
            sent.features['sent_nrc'] = emotions_vector

        return X


class Tense(Feature):
    """
    Adds the tense of the sentence as a feature.
    """
    print("Calculating ... Tense | features['tense']")
    FEATS = ['tense']
    TENSES = {'present': 0, "future": 1, "past": 2, 'pp': 3}

    def transform(self, X):
        for sent in X:
            tags = pos_tag(sent.tokens)
            sent.features['tense'] = self.TENSES['present']
            for tag in tags:
                if tag[1] in ['VBD']:
                    sent.features['tense'] = self.TENSES['past']
                if tag[0] in ['will']:
                    sent.features['tense'] = self.TENSES['future']
            if 'have' in sent.tokens and 'to' in sent.tokens:
                sent.features['tense'] = self.TENSES['future']

        return X


class SentimentLexicons(Feature):
    """
    Adds as feature the number of words in each sentence that appear in each of the sentiment lexicons.
    """
    print("Calculating ... SentimentLexicons (qatar lex) | features['lexicons']")
    FEATS = ['lexicons']

    def __init__(self):
        qatar_lexicon_files = ["negative-words-Liu05.txt", "negations.txt", "bias-lexicon-RecasensACL13.txt"]
        self.lexicons = []
        for lexicon in qatar_lexicon_files:
            self.lexicons.append(set(codecs.open(join(CONFIG['sentiment_lexicons'], lexicon), encoding='iso-8859-1').
                                     read().split("\n")))

    def transform(self, X):
        for sent in X:
            sent.features['lexicons'] = []
            for lexicon in self.lexicons:
                sent.features['lexicons'].append(sum([sent.tokens.count(lex_word) for lex_word in lexicon]))
        return X

# extracts mixed features / contradictions
class Negatives(Feature):
    """Adds negative words and contradictions counts in current and next sentence."""
    print("Calculating ... Negatives | features['negs','contras', 'negs_next','contras_next']")
    FEATS = ['negs', 'contras', 'negs_next', 'contras_next']

    def __init__(self):
        self.contras = get_contra_vocab()
        self.negatives = get_negative_vocab()

    def transform(self, X):
        for i, sent in enumerate(X):
            sent.features['negs'] = self.count_neg(sent.text)
            sent.features['contras'] = self.count_contras(sent.text)
            sent.features['negs_next'] = 0
            sent.features['contras_next'] = 0
            #if i < len(X) - 1 and sent.debate == X[i + 1].debate: (IQJ) (checking if there is a next sentence, and if it belongs to the same debate) X[i+1] means next sent.
            if i < len(X) - 1:
                sent.features['negs_next'] = self.count_neg(X[i + 1].text)
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


class NegationNextChunk(Negatives):
    """Adds negavie words count and contradictions in next chunk.
    Chunk is a sequence of sentences said by one person - term "segment" used in paper instead of "chunck"."""
    FEATS = ['negs_next_chunk', 'contras_next_chunk']

    def transform(self, X):
        for i, sent in enumerate(X):
            curr_chunk_i = i + 1
            while curr_chunk_i < len(X) and X[curr_chunk_i].speaker == X[i].speaker:
                curr_chunk_i += 1

            next_chunk_i = curr_chunk_i
            count_next_neg = 0
            count_next_contra = 0
            if next_chunk_i < len(X) and X[curr_chunk_i].debate == X[i].debate:
                while next_chunk_i < len(X) and X[next_chunk_i].speaker == X[curr_chunk_i].speaker:
                    count_next_neg += self.count_neg(X[next_chunk_i].text)
                    count_next_contra += self.count_contras(X[next_chunk_i].text)
                    next_chunk_i += 1
            sent.features['negs_next_chunk'] = count_next_neg
            sent.features['contras_next_chunk'] = count_next_contra
        return X
