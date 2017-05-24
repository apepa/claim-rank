from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler

from features import alchemy_feat, counting_feat, \
    dict_feat, metadata_feat, nltk_feat, topics, knn_similarity, embeddings_feat

from features.features import ToMatrix


def get_cb_pipeline(train):
    """
    Set of features used by Claim Buster.
    """
    features = [
        ('sentiment', alchemy_feat.Sentiment()),
        ('sent_len', counting_feat.SentenceLength()),
        ('tfidf', counting_feat.BagOfTfIDF(train)),
        ('ner', alchemy_feat.NER()),
        ('pos', nltk_feat.POS())
    ]
    return get_pipeline(features)


def get_experimential_pipeline(train):
    experimential_features = [
        ("pf_search", knn_similarity.PolitiFactSearch()),
        ("train_search", knn_similarity.TrainSearch(train=train)),
        ('emb_chunk', embeddings_feat.W2VVectorSim()),
        ('emb_sent', embeddings_feat.W2VVectors()),
        ('tense', dict_feat.Tense()),
        ('qatar_lex', dict_feat.SentimentLexicons()),
        ('speaker', metadata_feat.Speaker()),
        ('system', metadata_feat.System()),
        ('opponent', metadata_feat.TalkingAboutTheOther())
    ]
    return get_pipeline(experimential_features)


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
