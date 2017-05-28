from sklearn.pipeline import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler


class Feature(TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self

# this function extracts all the features of all the sentences and svaes them into one vector ready for SVM algorithm
class ToMatrix(Feature):
    """Transforms the features dict to a matrix"""
    def __init__(self, features=[]):
        self.features = features

    def transform(self, X):
        final_X = []

        for sent in X:
            sent_vector = []
            for feat in self.features: # if the features vector is of datatype list use the concatination to concatenate the arrays
                if isinstance(sent.features[feat], list):
                    sent_vector += sent.features[feat]
                else:                   # otherwise use the method append to concatenate
                    sent_vector.append(sent.features[feat])
            final_X.append(sent_vector)
        return final_X  # the final vector is the vector of features only (not a vector of sentence objects)

# X : numpy array of shape [n_samples, n_features] - Training set
# numpy array : multidimensional array   (read on numpy arrays) unlike built in python one dimensional arrays

def get_pipeline(features):
    """
    Constructs a pipeline with the given features.
    Adds dict to matrix of features transformer and a scaler.
    """
    feature_names = []
    for feature in features:
        feature_names += feature[1].FEATS
    print(feature_names)
    return Pipeline(features + [('transform', ToMatrix(features=feature_names)),
                                ('norm', MinMaxScaler())
                                ])
