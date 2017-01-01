from src.utils.config import *
CONFIG = get_config()

emotions = ["anger", "fear", "anticipation", "trust", "surprise", "sadness",
            "joy", "disgust", "positive", "negative"]


def get_stopwords():
    """
    :return: set (for fast searching) of the stopwords
    """
    stopwords_file = open(CONFIG['stopwords'])
    stopwords = set()
    for line in stopwords_file:
        stopwords.add(line.strip())
    return stopwords


def get_sentiment_lexicon():
    """
    Returns a dictionary with sentiments of words from NRC sentiment lexicon.
    :return: dictionary of sentiments, where the key is a word and the value is a dictionary of sentiments:
        'supported': {'anticipation': 0, 'trust': 0, 'surprise': 0, 'fear': 0,
            'sadness': 0, 'anger': 0, 'joy': 0, 'disgust': 0, 'negative': 0, 'positive': 1}
    """
    sentiment_file = open(CONFIG['sentiment_lex'])
    emotions_lex = {}
    for line in sentiment_file:
        line = line.strip()
        if "\t" not in line:
            continue
        columns = line.split("\t")
        if columns[0] not in emotions_lex:
            emotions_lex[columns[0]] = {}
        emotions_lex.get(columns[0])[columns[1]] = int(columns[2])
    return emotions_lex
