from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing.data import MinMaxScaler
import numpy as np
from features.feature_sets import get_cb_pipeline, get_serialized_pipeline, get_experimential_pipeline


def run(test, train):
    # generate features

    # feats = get_experimential_pipeline(train)
    feats = get_serialized_pipeline(train)

    train_x = feats.fit_transform(train)
    test_x = feats.transform(test)
    # train
    train_y = [1 if sent.label_test > 0 else 0 for sent in train]
    clf = MLPClassifier(max_iter=300, solver='sgd', alpha=4, hidden_layer_sizes=(200, 50),
                        random_state=42, activation='relu', learning_rate_init=0.03, batch_size=550)

    clf.fit(train_x, train_y)

    # predict
    predictions = clf.predict(test_x)
    pred_probs = clf.predict_proba(test_x)
    pred_probs = MinMaxScaler().fit_transform(np.reshape([pred[1] for pred in pred_probs], (-1, 1))).tolist()
    for i, sent in enumerate(test):
        sent.pred_label = predictions[i]
        sent.pred = pred_probs[i]

    return test
