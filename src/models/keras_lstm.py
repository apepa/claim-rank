from __future__ import print_function

from keras.layers.core import Dropout
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Merge

from features.feature_sets import get_experimential_pipeline
from src.utils.config import get_config

from os.path import join
import numpy as np
from src.data.debates import read_all_debates
from gensim.corpora import Dictionary

CONFIG = get_config()
dict = Dictionary([sent.tokens for sent in read_all_debates()])


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def feats_lstm_model(feat_size):
    text_embedding_model = Sequential()
    text_embedding_model.add(Embedding(5200, 100))
    text_embedding_model.add(Bidirectional(LSTM(200, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)))

    text_embedding_model.add(LSTM(100, dropout=0.7, recurrent_dropout=0.2, return_sequences=True))
    text_embedding_model.add(Dropout(0.5))
    text_embedding_model.add(LSTM(100, dropout=0.4, recurrent_dropout=0.2))
    text_embedding_model.add(Dropout(0.5))

    features_model = Sequential()
    features_model.add(Dense(300, activation='relu', input_shape=(feat_size,)))
    features_model.add(Dropout(0.1))
    features_model.add(Dense(100, activation='relu'))

    model = Sequential()
    model.add(Merge([text_embedding_model, features_model]))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=[precision]
                  )
    print(model.summary())
    return model


def lstm_model():
    model = Sequential()

    model.add(Embedding(6000, 200))
    model.add(Bidirectional(LSTM(200, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(200, dropout=0.8, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["binary_accuracy"]
                  )
    return model


def run(test, train, dump=False, use_dump=False):
    import numpy
    numpy.random.seed = 42
    batch_size = 32
    maxlen = 80

    features = get_experimential_pipeline(train)

    if use_dump:
        X_train_feats = np.load(join(CONFIG['feats_dump'], str(test[0].debate) + "_train.npy"))
        X_test_feats = np.load(join(CONFIG['feats_dump'], str(test[0].debate) + "_test.npy"))
    else:
        X_train_feats = features.fit_transform(train)
        X_test_feats = features.transform(test)

        if dump:
            np.save(str(test[0].debate) + "_test", X_test_feats)
            np.save(str(test[0].debate) + "_test", X_test_feats)

    y = [sent.label for sent in train]

    x_train = [[dict.token2id[token] for token in sent.tokens] for sent in train]
    y_train = [1 if sent.label > 0 else 0 for sent in train]
    x_test = [[dict.token2id[token] for token in sent.tokens] for sent in test]

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = feats_lstm_model(X_train_feats.shape[1])
    print('Train...')
    model.fit([x_train, X_train_feats], y_train,
              batch_size=batch_size,
              epochs=2,
              validation_data=([x_train, X_train_feats], y_train))

    out_prob = model.predict([x_test, X_test_feats])

    out = to_categorical(out_prob, num_classes=2)

    for i, sent in enumerate(test):
        sent.pred_label = out[i][0]
        sent.pred = out_prob[i][0]

    return test
