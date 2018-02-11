from os import listdir
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler
from src.features.features import ToMatrix, ReadFeatures
from src.utils.config import get_config


def get_cb_pipeline(train):
    """
    Set of features used by ClaimBuster.
    """
    from src.features import alchemy_feat, counting_feat, nltk_feat
    features = [
        ('sentiment', alchemy_feat.Sentiment()),
        ('sent_len', counting_feat.SentenceLength()),
        ('tfidf', counting_feat.BagOfTfIDF(train)),
        ('ner', alchemy_feat.NER()),
        ('pos', nltk_feat.POS())
    ]
    return get_pipeline(features)


def get_experimential_pipeline(train, to_matrix=True):
    from src.features import alchemy_feat, counting_feat, \
        dict_feat, metadata_feat, nltk_feat, topics, \
        knn_similarity, discourse, textblob_feat, embeddings_feat
    feats = [
        ("pf_search", knn_similarity.PolitiFactSearch()),
        ("train_search", knn_similarity.TrainSearch(train=train)),
        ('emb_chunk', embeddings_feat.W2VVectorSim()),
        ('w2v', embeddings_feat.W2VVectors()),
        ('tense', dict_feat.Tense()),
        ('qatar_lex', dict_feat.QatarLexicons()),
        ('sent', dict_feat.Sentiment_NRC()),
        ('ner_nltk', nltk_feat.NER()),
        ('sentiment', alchemy_feat.Sentiment()),  # cb
        ('speaker', metadata_feat.Speaker()),
        ('system', metadata_feat.System()),
        ('sent_len', counting_feat.SentenceLength()),  # cb
        ('ner', alchemy_feat.NER()),  # cb
        ('pos', nltk_feat.POS()),  # cb
        # ('syn', dict_feat.SyntacticParse()),
        # ('tfidfn', counting_feat.BagOfTfIDFN(train)),
        ('tfidf', counting_feat.BagOfTfIDF(train)),
        ('chunks', counting_feat.ChunkLen()),
        ('negatives', dict_feat.Negatives()),
        ('neg_chunk', dict_feat.NegationNextChunk()),
        ('lda', topics.LDATopics()),
        ('lda_sim', topics.LDAVectorSim()),
        # ('textblob', textblob_feat.TextBlobSentiment()),
        ('discourse', discourse.DiscourseInfo()),
        ('opponent', metadata_feat.TalkingAboutTheOther())
    ]
    return get_pipeline(feats, to_matrix)


def get_pipeline(features, to_matrix=True):
    """
    Constructs a pipeline with the given features.
    Adds dict to matrix of features transformer and a scaler.
    """
    feature_names = []
    for feature in features:
        feature_names += feature[1].FEATS
    if to_matrix:
        return Pipeline(features + [('transform', ToMatrix(features=feature_names)), ('norm', MinMaxScaler())])
    else:
        return Pipeline(features)


def get_serialized_pipeline(train):
    from src.features import counting_feat, knn_similarity
    config = get_config()

    black_list = ['polarity', 'subjectivity', 'sent_nrc',
                  'discourse_rel', 'discourse_it',
                  'in_chunk_last_it', 'in_chunk_last_rel', 'in_chunk_first_it', 'in_chunk_first_rel'
                  ]
    read_feature_names = [file_name for file_name in listdir(config['features_dump_dir']) if file_name not in black_list]

    all_feature_names = read_feature_names + counting_feat.BagOfTfIDFN.FEATS + knn_similarity.TrainSearch.FEATS
    print(all_feature_names)

    return Pipeline([('read', ReadFeatures(read_feature_names)),
                     ("train_search", knn_similarity.TrainSearch(train=train)),
                     ('tfidf', counting_feat.BagOfTfIDFN(train)),  # cb
                     ('transform', ToMatrix(features=all_feature_names)),
                     ('norm', MinMaxScaler())])
