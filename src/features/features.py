from sklearn.pipeline import TransformerMixin


class Feature(TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self


class ToMatrix(Feature):
    """Transforms the features dict to a matrix"""
    def __init__(self, features=[]):
        self.features = features

    def transform(self, X):
        final_X = []

        for sent in X:
            sent_vector = []
            for feat in self.features:
                if isinstance(sent.features[feat], list):
                    sent_vector += sent.features[feat]
                else:
                    sent_vector.append(sent.features[feat])
            final_X.append(sent_vector)
        return final_X

