from nltk.tokenize import sent_tokenize
#import random
from sklearn.svm import SVC
from features.feature_sets import *
#import pickle
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing.data import MinMaxScaler
from data.debates import *
from stats.rank_metrics import *


class mainclass ():
    #train = [] #global static class variable shared by all instances
    def __init__(self):

        return None
    # prepares user input to match the testing data format in the pipeline
    def mainproc(self,text,activateTraining,predictScores,method):
        sentenceList = self.tokenizeTranscript(text)
        self.sentenceVec = self.transTosentVec(sentenceList) # sentenceVec will be the vector (array) of sentence objects
        train = prepare_train_data_for_demo()
        if (activateTraining == True and method == 'svm' ):
            print ('TRAINING  SVM ============')
            results = self.prepare_data_getMetrics('svm')
        if (predictScores == True and method == 'svm'):
            self.sentenceVec = self.predict_scores_SVM(self.sentenceVec,train) # predict scores for the entered text (SVM)
        if (activateTraining == True and method == 'nn'):
            print ('TRAINING  NN  ============')
            results = self.prepare_data_getMetrics('nn')
        if (predictScores == True and method == 'nn'):
            self.sentenceVec = self.predict_scores_NN(self.sentenceVec,train)  # predict scores for the entered text (NN)

        return self.sentenceVec


    def tokenizeTranscript(self,text):
        sentenceList = sent_tokenize(text)    #tokenize transcript text into sentences
        return sentenceList

    def transTosentVec(self, sentenceList):
        sentenceVec = []
        i=0
        for sentence in sentenceList:
            sent = demoSentence(str(i),sentence) #creating a new instance from the class demoSentence and passing the constructor of the class the sentence text
            sentenceVec.append(sent)
            i+=1
        return sentenceVec

    # def generateScores(self, sentenceVec):
    #
    #     predictions = self.predict_scores_SVM(sentenceVec)
    #     return sentenceVec

    def prepare_data_getMetrics(self,method):
        results = []
        fold =0
        for test, train in get_for_crossvalidation():
            fold += 1
            print "F O L D   "+ str(fold) + " ##################################################################################"
            if method =='svm':
                self.trainModel_SVM(train)  # train the model with SVM (run only once)
                print ('TESTING  SVM ============')
                result = self.predict_scores_SVM(test,train) #test the model
            elif method =='nn':
                self.trainModel_NN(train)  # train the model with SVM (run only once)
                print ('TESTING  NN  ============')
                result = self.predict_scores_NN(test,train) #test the model
            results.append(result)
            print "FOLD "+str(fold)+" RESULTS : _________________________________________________________"
            get_all_metrics(results, agreement=1)
        print "FINAL RESULTS : __________________________________________________________"
        get_all_metrics(results, agreement=1)
        return results



########################################################################################################################################################################################
###############################################################  S U P P O R T   V E C T O R   M A C H I N E  ###########################################################################
####################################################################################################################################################################################



    def predict_scores_SVM(self,test,train):
        # PREDICTING RESULTS fOR DATA ENTERED BY THE USER
        agreement = 1
        # load the trained model saved as a python "pickle" file
        svc = joblib.load('trainedModelSVM.pkl')
        # calculate features
        features = get_demo_pipeline(train)
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

    def trainModel_SVM (self,train):

        """
        :param test:
        :param train:
        :param agreement:
        :return:
        """

        # this is the main gate (call) to the svm ranker (training, testing)
        #train = get_for_crossvalidation()
        agreement = 1
        C = 1
        gamma = 0.0001
        # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create SVM classification object
        svc = SVC(class_weight='balanced', kernel='rbf', C=C, gamma=gamma, probability=True, random_state=0)
        print ("Building features pipeline ...")
        features = get_demo_pipeline(train)
        # X_train is the list of training examples (features values / assembled vectors of features values)
        # X_train is an array of vectors (each vector represents an example from the training set )
        X_train = features.fit_transform(train)
        # y a parallel array to X_train holding the expected (value/class) for each example in X_train
        Y_train = [1 if sent.label >= agreement else 0 for sent in train]

        # TRAINING
        # Train the model using the training set
        print("Start training SVM.")
        svc.fit(X_train, Y_train)
        print("Finished training SVM.")

        # After training a scikit - learn model, it is desirable to have a way to persist the model for future use without having to retrain.
        # built-in Python serializaton function "Pickle" can be used for this purpose
        # In the specific case of the scikit, it may be more interesting to use joblib's replacement of pickle (joblib.dump & joblib.load),
        # which is more efficient on objects that carry large numpy arrays internally as is often the case for fitted scikit-learn estimators,
        # but can only pickle to the disk and not to a string
        joblib.dump(svc, 'trainedModelSVM.pkl')

#####################################################################################################################################################################################
################################################################  N E U R A L      N E T W O R K ####################################################################################
#####################################################################################################################################################################################

    # train the features with a Neural Net
    def trainModel_NN(self,train):

        #train = prepare_train_data_for_demo()

        print ("Building features pipeline ...")
        features = get_demo_pipeline(train)
        X_train = features.fit_transform(train)


        Y_train = [1 if sent.label > 0 else 0 for sent in train]
        clf = MLPClassifier(max_iter=100, solver='sgd', alpha=4, hidden_layer_sizes=(200, 50),
                            random_state=1, activation='relu', learning_rate_init=0.03, batch_size=550)
        # TRAINING
        # Train the model using the training set
        print("Start training NN.")
        clf.fit(X_train, Y_train)
        print("Finished training NN.")

        # preserve (pickle) the trained model
        joblib.dump(clf, 'trainedModelNN.pkl')


    # test with Neural Nets model
    def predict_scores_NN(self, test,train):

        # PREDICTING RESULTS fOR DATA ENTERED BY THE USER
        # load the trained model saved as a python "pickle" file
        clf = joblib.load('trainedModelNN.pkl')
        # calculate features
        features = get_demo_pipeline(train)
        X_test = features.fit_transform(test)

        predictions = clf.predict(X_test)
        pred_probs = clf.predict_proba(X_test)
        pred_probs = MinMaxScaler().fit_transform([pred[1] for pred in pred_probs]).tolist()


        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        return test
