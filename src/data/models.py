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

class demoSentence():
    def __init__(self, text, label=9):
        self.text = text
        #self.prediction = label
        self.label = label
        self.features = {}
        self.tokens = word_tokenize(text)


'''
class Sentence(object):
    def __init__(self,text,label=-1):
        #self.id = id
        self.text = text
        self.label =label                 #integer : level of agreement from the 9 resources
        #self.speaker = speaker
        #self.debate = debate
        self.features = {}
        #self.date = date
        self.tokens = word_tokenize(text)
'''