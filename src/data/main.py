from nltk.tokenize import sent_tokenize
import random
from data.models import *


class mainclass:
    def __init__(self):
        return None
    # prepares user input to match the testing data format in the pipeline
    def mainproc(self,text):
        sentenceList = self.tokenizeTranscript(text)
        self.sentenceVec = self.transTosentVec(sentenceList) # sentenceVec will be the vector (array) of sentence objects
        #self.trainModel() # train and test the model (run only once)
        self.sentenceVec = self.generateScores(self.sentenceVec) # predict scores for the entered text
        return self.sentenceVec


    def tokenizeTranscript(self,text):
        sentenceList = sent_tokenize(text)    #tokenize transcript text into sentences
        return sentenceList

    def transTosentVec(self, sentenceList):
        sentenceVec = []
        for sentence in sentenceList:
            sent = demoSentence(sentence) #creating a new instance from the class demoSentence and passing the constructor of the class the sentence text
            sentenceVec.append(sent)
        return sentenceVec

    def generateScores(self, sentenceVec):

        #for sentence in sentenceVec:
        #    sentence.score = round(random.uniform(0, 1),2)
        return sentenceVec


    def trainModel(self):
        from src.models.svm_cb import *
        from data.debates import *
        # this is the main gate (call) to the svm ranker (training, testing)
        for test_deb, test, train in get_for_crossvalidation(): # prepare 4 testing sets + for training sets (4-fold cross validation)
            dataset = run_svm_prob(test, train) # run the svm ranker on each of the training and testing sets form the cross validation and returns the train or test sets
            x = 0

