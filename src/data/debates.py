from enum import Enum
from os.path import join
from src.data.models import Sentence
from src.utils.config import get_config

CONFIG = get_config()
FILE_EXT = "_ann.tsv"
CB_FILE_EXT = '_cb.tsv'
SEP = "\t"


dabates_dates = {1: "2016-04-14T00:00:00",
                 2: "2016-09-25T00:00:00",
                 3: "2016-10-03T00:00:00",
                 4: "2016-10-08T00:00:00",
                 5: "2016-10-18T00:00:00",
                 6: "2016-07-21T00:00:00",
                 7: "2016-07-28T00:00:00",
                 8: "2017-01-20T00:00:00"}

class Debate(Enum):
    NinthDem = 1
    FIRST = 2
    VP = 3
    SECOND = 4
    THIRD = 5
    TRUMP_S = 6
    CLINTON_S = 7
    TRUMP_I = 8


DEBATES = [Debate.FIRST, Debate.VP, Debate.SECOND, Debate.THIRD]


def read_all_debates(source='ann'):
    """
    :param source:
    - 'ann' - annotations from different journalists' sources
    - 'cb' - label is the score from Claim Buster engine
    :return: a list of all sentences said in the debates
    """
    sentences = []
    if source == 'ann':
        sentences += read_debates(Debate.FIRST)
        sentences += read_debates(Debate.VP)
        sentences += read_debates(Debate.SECOND)
        sentences += read_debates(Debate.THIRD)

    elif source == 'cb':
        sentences += read_cb_scores(Debate.FIRST)
        sentences += read_cb_scores(Debate.VP)
        sentences += read_cb_scores(Debate.SECOND)
        sentences += read_cb_scores(Debate.THIRD)
    return sentences


def read_debates(debate, use_label='sum_all'):
    """
    Reads the debate transcripts data.
    :param debate: debates (Debate enum) to return the sentences for.
    :param use_label: how to form the gold label for the sentences
    - sum_all : label is the number of annotators that have agreed
    - lambda function : a custom function for a label, with input - the columns from file
    :return:

    Examples:
    1. Take for claims only those that more than one annotator agrees on it
    #>>> read_debates(Debate.FIRST, lambda x: 1 if int(x[2])>1 else 0)
    2. Take the number of annotators that have agreed on it
    #>>> read_debates(Debate.FIRST)
    """
    sentences = []
    debate_file_name = join(CONFIG['tr_all_anns'], CONFIG[debate.name] + FILE_EXT)
    debate_file = open(debate_file_name)
    debate_file.readline()
    for line in debate_file:
        line = line.strip()
        columns = line.split(SEP)

        if use_label == 'sum_all':
            label = int(columns[2].strip())
        else:
            label = use_label(columns)
        labels = columns[3:-1]
        s = Sentence(columns[0], columns[-1], label, columns[1], debate, dabates_dates[debate.value], labels)
        s.label_test = label
        sentences.append(s)

    return sentences


def read_cb_scores(debate):
    sentences = read_debates(debate)
    debate_file_name = join(CONFIG['tr_cb_anns'], CONFIG[debate.name] + CB_FILE_EXT)
    debate_file = open(debate_file_name)
    debate_file.readline()
    for i, line in enumerate(debate_file):
        line = line.strip()
        columns = line.split(SEP)
        sentences[i].pred = float(columns[2])
        sentences[i].pred_label = 1 if sentences[i].pred >= 0.5 else 0
    return sentences


def get_for_crossvalidation():
    """
    Splits the debates into four cross-validation sets.
    One of the debates is a test set at each cross validation.
    :return: test and train sets
    """
    data_sets = []
    for i, debate in enumerate(DEBATES):
        train_debates = DEBATES[:]
        train_debates.pop(i)
        train = []
        test = read_debates(debate)
        for train_debate in train_debates:
            train += read_debates(train_debate)
        data_sets.append((debate, test, train))
    return data_sets
