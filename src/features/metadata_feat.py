from src.features.features import Feature


class Speaker(Feature):
    """Adds the speaker of the sentence as a feature."""
    FEATS = ['speaker']

    def __init__(self):
        self.speakers = {}

    def transform(self, X):
        for sent in X:
            if sent.speaker not in self.speakers:
                self.speakers[sent.speaker.strip()] = len(self.speakers)
            sent.features['speaker'] = self.speakers[sent.speaker.strip()]
        return X


class System(Feature):
    """Adds indication whether the next chunk contains audience reaction or crosstalk."""
    FEATS = ['(laugh', '(crosstalk', '(applause', '(laughter']

    def transform(self, X):
        for i, sent in enumerate(X):
            for feat in self.FEATS:
                if i + 1 < len(X) and feat in X[i + 1].text.lower():
                    sent.features[feat] = 1
                else:
                    sent.features[feat] = 0
        return X
