from sklearn.svm import SVC
from sklearn.metrics.ranking import average_precision_score
from src.features.feature_sets import *
from random import shuffle, randint


def run_svm_prob(test, train, agreement=1, C=1, gamma=0.0001):
    """
    :param test:
    :param train:
    :param agreement:
    :return:
    """
    svc = SVC(class_weight='balanced', kernel='rbf', C=C, gamma=gamma, probability=True, random_state=0)
    features = get_experimential_pipeline(train)
    X_train = features.fit_transform(train)

    y = [1 if sent.label >= agreement else 0 for sent in train]

    print("Start training SVM.")
    svc.fit(X_train, y)
    print("Finished training SVM.")
    X = features.fit_transform(test)
    y_pred_proba = svc.predict_proba(X)
    y_pred_proba = MinMaxScaler().fit_transform([pred[1] for pred in y_pred_proba]).tolist()

    y_pred = svc.predict(X)

    for sent, prob, pred_label in zip(test, y_pred_proba, y_pred):
        sent.pred = prob
        sent.pred_label = pred_label

    y_true = [1 if s.label >= agreement else 0 for s in test]

    print(average_precision_score(y_true, y_pred_proba))
    return test


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

    features = get_experimential_pipeline(train)
    X_train = features.fit_transform(train)

    y = [1 if sent.label >= agreement else 0 for sent in train]
    X_train, y = balance(X_train, y)

    print("Start training SVM.")
    svc.fit(X_train, y)
    print("Finished training SVM.")
    X = features.fit_transform(test)

    y_pred_proba = svc.decision_function(X)
    y_pred_proba = MinMaxScaler().fit_transform(y_pred_proba).tolist()

    y_pred = svc.predict(X)

    for sent, prob, pred_label in zip(test, y_pred_proba, y_pred):
        sent.pred = prob
        sent.pred_label = pred_label

    y_true = [1 if s.label >= agreement else 0 for s in test]

    print(average_precision_score(y_true, y_pred_proba))
    return test


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
