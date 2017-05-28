
import subprocess
from utils.config import *
from data.debates import get_for_crossvalidation
from data.svm_converter import save_for_svm_rank, read_svm_pred
from features.feature_sets import get_cb_pipeline, get_experimential_pipeline
# from features.cb_features import get_pipeline
from os.path import join

CONFIG = get_config()

def run_svm_rank_crossval(C=1):
    test_res = []
    for debate_name, test, train in get_for_crossvalidation():
        test_res += run_svm_rank(train, test, new_features=True, C=C)
    return test_res

def run_svm_rank(sentences_train, sentences_test, new_features=False, C=3, test_name=None):
    """
    Calls command line the svm_rank classifier.
    :param new_features: whether to generate new features or use already generated.
    :param C: C param for SVM alg.
    :return: test instances sorted by their rank, given by svm_rank alg.

    >>>print_results(run_svm_rank(new_features=True, C=1.5))
    """

    if new_features:
        generate_new_features(sentences_test, sentences_train)

    if test_name is None:
        test_name = CONFIG['svm_rank_pred']
    else:
        test_name = join(CONFIG['svm_rank'], test_name)

    run_cmd("{} -c {} {} {}".format(CONFIG['svm_rank_learn'], C,
                                    CONFIG['svm_rank_train'], CONFIG['svm_rank_model']))

    run_cmd("{} {} {} {}".format(CONFIG['svm_rank_classify'], CONFIG['svm_rank_test'],
                                 CONFIG['svm_rank_model'], test_name,))

    results = read_svm_pred(sentences_test, test_name)
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
    pipeline = get_experimential_pipeline(sentences_train)

    X = pipeline.fit_transform(sentences_train)
    y = [sent.label for sent in sentences_train]
    save_for_svm_rank(X, y, CONFIG['svm_rank_train'])

    X_test = pipeline.transform(sentences_test)
    save_for_svm_rank(X_test, [], CONFIG['svm_rank_test'])
