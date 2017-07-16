
import textblob
from src.features.features import Feature


class TextBlobSentiment(Feature):

    FEATS = ['polarity', 'subjectivity']
    def transform(self, X):
        print("Calculating ... TextBlobSentiment | features['polarity','subjectivity']")
        for sent in X:
            blob = textblob.TextBlob(sent.text)
            sent.features['polarity'] = blob.sentiment.polarity
            sent.features['subjectivity'] = blob.sentiment.subjectivity
        return X