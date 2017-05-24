import mord
import numpy as np
from os.path import join

from utils.config import get_config
from features.feature_sets import get_experimential_pipeline

CONFIG = get_config()


def main(test, train, dump=False, use_dump=False):
    features = get_experimential_pipeline(train)

    if use_dump:
        X_train = np.load(join(CONFIG['feats_dump'], str(test[0].debate) + "_train.npy"))
        X_test = np.load(join(CONFIG['feats_dump'], str(test[0].debate) + "_test.npy"))
    else:
        X_train = features.fit_transform(train)
        X_test = features.transform(test)

        if dump:
            np.save(str(test[0].debate) + "_test", X_train)
            np.save(str(test[0].debate) + "_test", X_test)

    y = [sent.label for sent in train]

    print("Training...")
    clf = mord.OrdinalRidge(alpha=1)
    clf.fit(X_train, y)

    y_pred = clf.predict(X_test)

    for i, sent in enumerate(test):
        sent.pred_label =y_pred[i]
        sent.pred =y_pred[i]

    return test