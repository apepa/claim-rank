from operator import attrgetter
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score
from src.data.debates import get_for_crossvalidation, DEBATES, read_debates
from src.models.svm_rank import run_svm_rank
from operator import itemgetter
from src.data.svm_converter import read_svm_pred
import numpy as np


def precision_at_n(dataset, n=10):
    """
    Return precision of first n top ranked results.
    :param dataset: Sentences to get PR@N for
    :param n: number of top sentences to measure precision for.
    :return: PR@N
    """
    dataset = sorted(dataset, key=attrgetter('pred'), reverse=True)
    relevant = sum([1 if instance.label >= 1 else 0 for instance in dataset[:n]])
    return relevant/n

def get_mrr(sentences):
    mrr = 0
    sorted_res = sorted(sentences, key=attrgetter("pred"), reverse=True)
    for i, res in enumerate(sorted_res):
        if res.label >= 1:
            mrr += 1/(i+1)
    return mrr

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def get_all_metrics(sentences):
    """
    Prints list of results for a ranker.
    :param sentences:
    :return:
    """
    print(len(sentences))
    metrics = {'RR': [], 'AvgP': [], 'ROC': [],
               'PR@1': [], 'PR@3': [], 'PR@5': [], 'PR@10': [], 'PR@100': [], 'PR@200': []}

    for sentence_set in sentences:
        y_true = [1 if t.label >= 1 else 0 for t in sentence_set]
        y_pred = [s.pred for s in sentence_set]

        metrics['AvgP'].append(average_precision_score(y_true, y_pred))
        metrics['ROC'].append(roc_auc_score(y_true, y_pred))
        metrics['RR'].append(get_mrr(sentence_set))

        for i in [1, 3, 5, 10, 100, 200]:
            metrics['PR@'+str(i)].append(precision_at_n(sentence_set, i))

    for key, value in sorted(metrics.items(), key=itemgetter(0)):
        print("{0}\t\t {1:.2f}".format(key, mean(value)))

def get_metrics_for_plot(agreement, ranks):
    """
    Calculates the precision, recalls and avg. precisions at various threshold levels.
    :param agreement: level of agreement for a cliam to be counted as positive
    :param ranks: ranks given by a classifier/ranker
    :return: average precisions, precisions, recalls, thresholds
    """
    precision = []
    recall = []
    av_p = []

    step = 0.01
    thresholds = [i for i in np.arange(0, 1 + step, step)]

    for threshold in thresholds:
        av_p_th = []
        precision_th = []
        recall_th = []

        for debate_sents in ranks:
            # get for positives claims with agreement above 'agreement'
            y_score = [1 if sent.label >= agreement else 0 for sent in debate_sents]

            # get for positive predictions those with rank above threshold
            y_test = [1 if sent.pred > threshold else 0 for sent in debate_sents]
            precision_th.append(precision_score(y_true=y_score, y_pred=y_test))
            recall_th.append(recall_score(y_true=y_score, y_pred=y_test))
            av_p_th.append(average_precision_score(y_true=y_score, y_score=y_test))

        av_p.append(mean(av_p_th))
        precision.append(mean(precision_th))
        recall.append(mean(recall_th))

    return av_p, precision, recall, thresholds

if __name__ == '__main__':
    """
    results = []
    for test_deb, test, train in get_for_crossvalidation():
        results.append(run_svm_rank(train, test, new_features=True, test_name=test_deb.name))
    get_all_metrics(results)
    """

    results = []
    from src.utils.config import get_config
    from os.path import join
    CONFIG = get_config()
    for debate in DEBATES:
        results.append(read_svm_pred(read_debates(debate), join(CONFIG['svm_rank']+debate.name)))
    get_all_metrics(results)