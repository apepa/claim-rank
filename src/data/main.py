from nltk.tokenize import sent_tokenize
#import random
from data.models import *
#from sklearn.svm import SVC
from features.feature_sets import *
#import pickle
from sklearn.externals import joblib

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

        predictions = self.run_demo_svm(sentenceVec)
        return sentenceVec


    def trainModel(self):
        from src.models.svm_cb import *
        from data.debates import *
        # this is the main gate (call) to the svm ranker (training, testing)
        for test_deb, test, train in get_for_crossvalidation(): # prepare 4 testing sets + for training sets (4-fold cross validation)
            trained_ds = run_svm_prob(test, train) # run the svm ranker on each of the training and testing sets form the cross validation and returns the train or test sets
            x = 0

    def run_demo_svm(self,test):
        # PREDICTING RESULTS fOR DATA ENTERED BY THE USER
        agreement = 1
        # load the trained model saved as a python "pickle" file
        svc = joblib.load('trainedModel.pkl')
        # calculate features
        features = get_cb_pipeline(test)
        X = features.fit_transform(test)
        y_pred_proba = svc.predict_proba(X)
        y_pred_proba = MinMaxScaler().fit_transform([pred[1] for pred in y_pred_proba]).tolist()
        # Predict Output for the "end-user input" dataset
        y_pred = svc.predict(X)

        # now after getting the predictions and the prob. of the predictions , attach these values to each sentence object in the testing set
        # in other words, annotate the testing set with the predicted values and the prob.s
        for sent, prob, pred_label in zip(test, y_pred_proba, y_pred):
            sent.pred = prob
            sent.pred_label = pred_label

        y_true = [1 if s.label >= agreement else 0 for s in test]

        #print(average_precision_score(y_true, y_pred_proba))
        return test  # return annotated data set ( array of sentence objects but with sentence.label values filled)