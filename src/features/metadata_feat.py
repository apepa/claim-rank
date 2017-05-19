from src.features.features import Feature


class TalkingAboutTheOther(Feature):
    """
    Indicates whether the participant is talking about the opponent and
    whether he is an opponent or a moderator.
    """

    MODERATORS = ['QUIJANO', 'COOPER', 'QUESTION', 'HOLT', 'WALLACE', 'RADDATZ']
    OPPONENTS = {'CLINTON': ['Donald', 'Trump', 'Pence', 'Mike'],
                 'TRUMP': ['Kaine', 'Hillary', 'Clinton', 'Tim'],
                 'PENCE': ['Clinton', 'Hillary', 'Kaine', 'Tim'],
                 'KAINE': ['Donald', 'Trump', 'Pence', 'Mike']}

    FEATS = ['participant', 'opponent']

    def transform(self, X):
        for sent in X:
            talking_about_opponent = False
            is_participant = False

            speaker = sent.speaker.strip()

            if speaker not in self.MODERATORS:
                is_participant = True

            if speaker in self.OPPONENTS and \
                    any(opponent in sent.text for opponent in self.OPPONENTS[speaker]):
                    talking_about_opponent = True

            sent.features['opponent'] = int(talking_about_opponent)
            sent.features['participant'] = int(is_participant)
        return X


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
