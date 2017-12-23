import json
from sklearn.pipeline import TransformerMixin
from src.utils.config import get_config


class Feature(TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self


class ReadFeatures(Feature):
    def __init__(self, feature_names):
        config = get_config()
        self._feature_dicts = {}
        for feature in feature_names:
            self._feature_dicts[feature] = json.loads(open(config['features_dump_dir']+feature).read())

    def transform(self, X):
        for _x in X:
            for feature_name, feature_values in self._feature_dicts.items():
                _x.features[feature_name] = feature_values[str(_x.id)+_x.debate.name]
        return X


class ToMatrix(Feature):
    """Transforms the features dict to a matrix"""
    def __init__(self, features):
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

