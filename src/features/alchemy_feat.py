from src.features.features import Feature
from src.utils.config import get_config

CONFIG = get_config()

# alchemy API is used o extract sentence level features / ClaimBuster based features : named entities (NER = Named Entities Recognition)
class NER(Feature):
    """Adds Named Entities from Alchemy API - [ 20 f ]  """
    FEATS = ['ner']

    def __init__(self):
        types_file = open(CONFIG['alchemy_ner'])
        self.ner = {}
        columns = []
        types_file.readline()
        for line in types_file:
            line = line.strip()
            columns = line.split("\t")
            self.ner[columns[0] + " " + columns[1]] = [int(v) for v in columns[2:]]
        self.len = len(columns) - 2

    def transform(self, X):
        for sent in X:
            sent.features['ner'] = self.ner[str(sent.id) + " " + sent.debate.name]
        return X

#extracts Sentence-level features/claimBuster based : sentiment
class Sentiment(Feature):
    """Adds Sentiment score from Alchemy API."""
    SENTIMENT_INDEX = 3
    DEBATE_INDEX = 1
    ID_INDEX = 0
    FEATS = ['sent']

    def __init__(self):
        self.sentiment = {}

        # read serialized sentiment scores
        serialized_sentiment = open(CONFIG['sentiment_file'])
        serialized_sentiment.readline()
        for line in serialized_sentiment:
            line = line.strip()
            columns = line.split("\t")
            if columns[self.SENTIMENT_INDEX] == 'Type':
                continue
            self.sentiment[columns[self.DEBATE_INDEX] + columns[self.ID_INDEX]] = float(columns[self.SENTIMENT_INDEX])

    def transform(self, X):
        for sent in X:
            sent.features['sent'] = self.sentiment[sent.debate.name + sent.id]
        return X
