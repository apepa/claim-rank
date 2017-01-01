import subprocess
from src.utils.config import *
from src.data.debates import Debate
from src.data.debates import read_debates
from src.data.svm_converter import save_for_svm_rank, read_svm_pred
from src.features.features import get_pipeline

CONFIG = get_config()

debates = [Debate.FIRST, Debate.VP, Debate.SECOND, Debate.THIRD]


def get_datasets():
    sentences_train = []
    for i in range(len(debates)-1):
        sentences_train += read_debates(debates[i])
    sentences_test = read_debates(debates[-1])

    return sentences_train, sentences_test


def run_svm_rank(new_features=False, C=3):
    """
    Calls command line the svm_rank classifier.
    :param new_features: whether to generate new features or use already generated.
    :param C: C param for SVM alg.
    :return: test instances sorted by their rank, given by svm_rank alg.

    >>>print_results(run_svm_rank(new_features=True, C=1.5))
    """

    sentences_train, sentences_test = get_datasets()

    if new_features:
        generate_new_features(sentences_test, sentences_train)

    run_cmd("{} -c {} {} {}".format(CONFIG['svm_rank_learn'], C,
                                    CONFIG['svm_rank_train'], CONFIG['svm_rank_model']))

    run_cmd("{} {} {} {}".format(CONFIG['svm_rank_classify'], CONFIG['svm_rank_test'],
                                 CONFIG['svm_rank_model'], CONFIG['svm_rank_pred'],))

    results = read_svm_pred(sentences_test, CONFIG['svm_rank_pred'])
    return results


def print_results(results):
    for sent in results:
        print("{} {} {}".format(sent.label, sent.rank, sent.text))

def run_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    print(error)


def generate_new_features(sentences_test, sentences_train):
    pipeline = get_pipeline(sentences_train)

    X = pipeline.transform(sentences_train)
    y = [sent.label for sent in sentences_train]
    save_for_svm_rank(X, y, CONFIG['svm_rank_train'])

    X_test = pipeline.transform(sentences_test)
    save_for_svm_rank(X_test, [], CONFIG['svm_rank_test'])
