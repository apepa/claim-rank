from nltk.tokenize import sent_tokenize
#import random
from data.models import *
from sklearn.svm import SVC
from features.feature_sets import *
#import pickle
from sklearn.externals import joblib

class mainclass:
    def __init__(self):
        return None
    # prepares user input to match the testing data format in the pipeline
    def mainproc(self,text,activateTraining,predictScores):
        sentenceList = self.tokenizeTranscript(text)
        self.sentenceVec = self.transTosentVec(sentenceList) # sentenceVec will be the vector (array) of sentence objects
        if(activateTraining== True):
            self.trainModel() # train and test the model (run only once)
        if(predictScores == True):
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

        predictions = self.predict_scores(sentenceVec)
        return sentenceVec


    def predict_scores(self,test):
        # PREDICTING RESULTS fOR DATA ENTERED BY THE USER
        agreement = 1
        # load the trained model saved as a python "pickle" file
        svc = joblib.load('trainedModel.pkl')
        # calculate features
        features = get_demo_pipeline(test)
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

        #y_true = [1 if s.label >= agreement else 0 for s in test]

        #print(average_precision_score(y_true, y_pred_proba))
        return test  # return annotated data set ( array of sentence objects but with sentence.label values filled)

    def trainModel(self):

        """
        :param test:
        :param train:
        :param agreement:
        :return:
        """
        from data.debates import *
        # this is the main gate (call) to the svm ranker (training, testing)
        train = prepare_train_data_for_demo()
        agreement = 1
        C = 1
        gamma = 0.0001
        # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create SVM classification object
        svc = SVC(class_weight='balanced', kernel='rbf', C=C, gamma=gamma, probability=True, random_state=0)
        features = get_demo_pipeline(train)
        # X_train is the list of training examples (features values / assembled vectors of features values)
        # X_train is an array of vectors (each vector represents an example from the training set )
        X_train = features.fit_transform(train)
        # y a parallel array to X_train holding the expected (value/class) for each example in X_train
        y = [1 if sent.label >= agreement else 0 for sent in train]

        # TRAINING
        # Train the model using the training set
        print("Start training SVM.")
        svc.fit(X_train, y)
        print("Finished training SVM.")

        # After training a scikit - learn model, it is desirable to have a way to persist the model for future use without having to retrain.
        # built-in Python serializaton function "Pickle" can be used for this purpose
        # In the specific case of the scikit, it may be more interesting to use joblib's replacement of pickle (joblib.dump & joblib.load),
        # which is more efficient on objects that carry large numpy arrays internally as is often the case for fitted scikit-learn estimators,
        # but can only pickle to the disk and not to a string
        joblib.dump(svc, 'trainedModel.pkl')

