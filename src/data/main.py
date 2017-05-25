from nltk.tokenize import sent_tokenize
import random
from src.data.models import *
class mainclass:
    def __init__(self):
        return None
    # prepares user input to match the testing data format in the pipeline
    def mainproc(self,text):
        sentenceList = self.tokenizeTranscript(text)
        self.sentenceVec = self.transTosentVec(sentenceList)
        self.sentenceVec = self.generateScores(self.sentenceVec)
        return self.sentenceVec


    def tokenizeTranscript(self,text):
        sentenceList = sent_tokenize(text)    #tokenize transcript text into sentences
        return sentenceList

    def transTosentVec(self, sentenceList):
        sentenceVec = []
        for sentence in sentenceList:
            sent = Sentence()
            sent.text = sentence
            sentenceVec.append(sent)
        return sentenceVec

    def generateScores(self, sentenceVec):
        for sentence in sentenceVec:
            sentence.score = round(random.uniform(0, 1),2)
        return sentenceVec

