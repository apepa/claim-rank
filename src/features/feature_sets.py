# this file functions are to construct the pipeline from the features
# first function constructs a pipeline from the claim buster features,
# second function constructs a pipeline form the proposed feature set


from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler

from features import alchemy_feat, counting_feat, dict_feat, metadata_feat, nltk_feat, topics, knn_similarity, embeddings_feat
from features.features import ToMatrix


def get_cb_pipeline(train):
    """
    Set of features used by Claim Buster.
    """
    features = [
        ('sentiment', alchemy_feat.Sentiment()),   # 1f
        ('sent_len', counting_feat.SentenceLength()),   # 1f no of tokens
        ('tfidf', counting_feat.BagOfTfIDF(train)),     # 998f TF-IDF weighted BOW
        ('ner', alchemy_feat.NER()),    # 20f named entities (alchemy)
        ('pos', nltk_feat.POS())    # 25f POS tags
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

# this function constructs a pipeline from the features that can be put into a fast demo (IQJ)
def get_demo_pipeline(train):
    demo_features = [
        ('',),
        ('',),
        ('',),
        ('',),
        ('',),
        ('',),
        ('',),
        ('',),
    ]
    return get_pipeline(demo_features)

# this function takes a training set (features), and returns a pipeline of features ready to be transformed and trained
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

#  The Pipeline is built using a list of (key, value) pairs,
#  where the key is a string containing the name you want to give this step and value is an estimator object