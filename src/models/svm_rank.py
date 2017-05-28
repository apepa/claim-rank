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

def run_svm_rank_crossval(C=1):
    test_res = []
    for i, debate in enumerate(debates):
        test_sents = read_debates(debate)
        train_sents = []
        temp_debates = debates[:]
        temp_debates.pop(i)
        for train_debate in temp_debates:
            train_sents += read_debates(train_debate)
        test_res += run_svm_rank(sentences_train=train_sents, sentences_test=test_sents, new_features=True, C=C)
    return test_res


# This is the main function responsible for training a model (classifier), and testing the model with testing data, then produce predictions file
def run_svm_rank(sentences_train, sentences_test, new_features=False, C=3):
    """
    Calls command line the svm_rank classifier.
    :param new_features: whether to generate new features or use already generated.
    :param C: C param for SVM alg.
    :return: test instances sorted by their rank, given by svm_rank alg.

    >>>print_results(run_svm_rank(new_features = True, C = 1.5))
    """


    if new_features:
        generate_new_features(sentences_test, sentences_train)

    # runs the SVM_Rank algorithm command to learn a classifier from the training data
    run_cmd("{} -c {} {} {}".format(CONFIG['svm_rank_learn'], C,
                                    CONFIG['svm_rank_train'], CONFIG['svm_rank_model']))

    # runs a command that uses the classifier learned above to classify the testing data (produce prediction scores for the testing examples)
    run_cmd("{} {} {} {}".format(CONFIG['svm_rank_classify'], CONFIG['svm_rank_test'],
                                 CONFIG['svm_rank_model'], CONFIG['svm_rank_pred'],))

    # read the predictions produced above and save them in the variable results
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

# CONFIG['something'] means get the value from the config.ini file for the variable named 'something'
# X the training set (array of sentences with their features )
# Y classification labels (array of classification labels) for each sentence from X