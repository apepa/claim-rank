from keras import backend as K
import os

import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import average_precision_score

def init_session(use_gpu=True):
    random.seed(1)
    os.environ['PYTHONHASHSEED'] = '1'
    tf.set_random_seed(1)
    np.random.seed(1)
    K.get_session().close()
    cfg = K.tf.ConfigProto()

    if not(use_gpu):
        cfg = K.tf.ConfigProto(device_count = {'GPU': 0})

    cfg.gpu_options.allow_growth = True
    cfg.inter_op_parallelism_threads = 1
    cfg.intra_op_parallelism_threads = 1

    K.set_session(K.tf.Session(config=cfg))

def cross_validate(folds, train_targets, keras_model):
  models = []

  for fold in folds:
      X_train, _, y_train, _ = fold
      y_train = list(y_train[:, train_targets].T)
      model = keras_model.fit(X_train, y_train)
      models.append(model)

  return models

def get_score_multitask(folds, models, test_targets, output_indexes):
  ap = []
  for fold, model in zip(folds, models):
      _, X_test, _, y_test = fold
      y_test = list(y_test[:, test_targets].T)
      ap.append([average_precision_score(y_test[j], model.predict(X_test)[i][:, 0])
            for j, i in enumerate(output_indexes)])

  return np.sum(ap, 0) / len(folds)


def get_score_singletask(folds, models, test_target):
  ap = []
  for fold, model in zip(folds, models):
      _, X_test, _, y_test = fold
      y_test = y_test[:, test_target]
      ap.append(average_precision_score(y_test, model.predict(X_test)[:, 0]))

  return np.sum(ap, 0) / len(folds)