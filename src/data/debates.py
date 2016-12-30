from enum import Enum
from os.path import join
from src.data.models import Sentence
from src.utils.config import get_config

CONFIG = get_config()
FILE_EXT = "_ann.tsv"
SEP = "\t"


class Debate(Enum):
    FIRST = 1
    VP = 2
    SECOND = 3
    THIRD = 4


def read_debates(debates, use_label = 'sum_all'):
    """
    Reads the debate transcripts data.
    :param debates: list of the debates(Debate enums) to return the sentences for.
    :param use_label: how to form the gold label for the sentences
    - sum_all : label is the number of annotators that have agreed
    - lambda function : a custom function for a label, with input - the columns from file
    :return:

    Examples:
    1. Take for claims only those that more than one annotator agrees on it
    >>> read_debates([Debate.FIRST], lambda x: 1 if int(x[2])>1 else 0)
    2. Take the number of annotators that have agreed on it
    >>> read_debates([Debate.FIRST])
    """
    sentences = []
    for debate in debates:
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

            s = Sentence(columns[0], columns[-1], label, columns[1], debate)
            sentences.append(s)

    return sentences

