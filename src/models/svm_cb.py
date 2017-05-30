from sklearn.svm import SVC
from sklearn.metrics.ranking import average_precision_score
from features.feature_sets import *
from random import shuffle, randint

# test and train are arrays of "sentence objects" of the testing and training sets
def run_svm_prob(test, train, agreement=1, C=1, gamma=0.0001):
    """
    :param test:
    :param train:
    :param agreement:
    :return:
    """
    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create SVM classification object
    svc = SVC(class_weight='balanced', kernel='rbf', C=C, gamma=gamma, probability=True, random_state=0)
    features = get_cb_pipeline(train)
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

    # TESTING
    X = features.fit_transform(test)
    y_pred_proba = svc.predict_proba(X)
    y_pred_proba = MinMaxScaler().fit_transform([pred[1] for pred in y_pred_proba]).tolist()
    # Predict Output for the testing set
    y_pred = svc.predict(X)

    # now after getting the predictions and the prob. of the predictions , attach these values to each sentence object in the testing set
    # in other words, annotate the testing set with the predicted values and the prob.s
    for sent, prob, pred_label in zip(test, y_pred_proba, y_pred):
        sent.pred = prob
        sent.pred_label = pred_label

    y_true = [1 if s.label >= agreement else 0 for s in test]

    print(average_precision_score(y_true, y_pred_proba))
    return test # return annotated testing set ( array of sentence objects but with sentence.label values filled)

# decision distance refers to the distance between the point and the hyperplane (hyperplane margins)
# the difference between this function and the baove function is only one thing
# this functions eastimates the classificaiton prob by using the decision distance
# the above function uses the regular way (different way) to estimate the prob
def run_svm_decision_distance(test, train, agreement=1):
    """
    :param test:
    :param train:
    :param agreement:
    :return:
    """
    from sklearn.pipeline import Pipeline
    svc = Pipeline([
        ("svm", SVC(class_weight='balanced', kernel='rbf', C=0.7, gamma=0.001, random_state=0))])

    features = get_cb_pipeline(train)
    X_train = features.fit_transform(train)

    y = [1 if sent.label >= agreement else 0 for sent in train]
    X_train, y = balance(X_train, y)

    print("Start training SVM.")
    svc.fit(X_train, y)
    print("Finished training SVM.")
    X = features.fit_transform(test)

    y_pred_proba = svc.decision_function(X) # estimate the prob. from the distance between the point and the hyperplane (margins)
    y_pred_proba = MinMaxScaler().fit_transform(y_pred_proba).tolist()

    y_pred = svc.predict(X)


    for sent, prob, pred_label in zip(test, y_pred_proba, y_pred):
        sent.pred = prob
        sent.pred_label = pred_label

    y_true = [1 if s.label >= agreement else 0 for s in test]

    print(average_precision_score(y_true, y_pred_proba))
    return test

# creates a new training set of equal number of positive and negative examples (balanced training set)
# allows for the hyperplane to adjust more correctly (get a better hyperplane when the training set is balanced)
def balance(X_train, y, scale='up'):
    print("Start balancing.")
    train_0 = [i for i, t in enumerate(y) if t == 0]
    train_1 = [i for i, t in enumerate(y) if t == 1]

    newtrain = train_0[:]

    num_train_1 = 0
    while num_train_1 < len(train_0):
        newtrain.append(train_1[randint(0, len(train_1)-1)])
        num_train_1 += 1

    shuffle(newtrain)
    newX = []
    newy = []
    for i in newtrain:
        newX.append(X_train[i])
        newy.append(y[i])
    print("Done balancing.")
    return newX, newy
