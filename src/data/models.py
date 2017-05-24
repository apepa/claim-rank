from nltk.tokenize import word_tokenize

class Sentence(object):
    def __init__(self, id, text, label, speaker, debate, date):
        self.id = id
        self.text = text
        self.label = label
        self.speaker = speaker
        self.debate = debate
        self.features = {}
        self.date = date
        self.tokens = word_tokenize(text)
