from __future__ import division
from operator import attrgetter
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score
from src.utils.config import get_config
from operator import itemgetter
import numpy as np
from math import *
from copy import deepcopy

#from data.debates import get_for_crossvalidation
#from src.models.sklearn_nn import run

CONFIG = get_config()


def precision(y_true, y_pred):
    print(y_pred)
    print(y_true)
    tp = sum([1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1])
    fp = sum([1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1])
    return tp/tp+fp


def r_precision(dataset, agreement=1):
    R = sum([1 if sent.label >= agreement else 0 for sent in dataset])
    return precision_at_n(dataset, n=R, agreement=agreement)


def f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def accuracy(y_true, y_pred):
    num_correct = len([1 for true, pred in zip(y_true, y_pred) if true == pred])
    return num_correct/len(y_true)


def average_precision(dataset, agreement):
    sorted_dataset = sorted(dataset, key=attrgetter("pred"), reverse=True)
    relevant = sum([1 if i.label >= agreement else 0 for i in dataset])
    avg_p = 0
    for i, inst in enumerate(sorted_dataset):
        if inst.label >= agreement:
            avg_p += precision_at_n(dataset, n=i+1, agreement=agreement)
    return avg_p/relevant


def precision_at_n(dataset, n=10, agreement=1):
    """
    Return precision of first n top ranked results.
    :param dataset: Sentences to get PR@N for
    :param n: number of top sentences to measure precision for.
    :return: PR@N
    """
    dataset = sorted(dataset, key=attrgetter('pred'), reverse=True)
    relevant = sum([1 if instance.label >= agreement else 0 for instance in dataset[:n]])
    p=relevant/n
    return p


def recall_at_n(dataset, n=10, agreement=1):
    dataset = sorted(dataset, key=attrgetter('pred'), reverse=True)
    relevant = sum([1 if instance.label >= agreement else 0 for instance in dataset[:n]])
    all_relevant = sum([1 if instance.label >= agreement else 0 for instance in dataset])
    return relevant / all_relevant


def dcg(dataset, agreement=True, agreement_num=1):
    dataset = sorted(dataset, key=attrgetter('pred'), reverse=True)
    result = 0
    for i, instance in enumerate(dataset):
        if agreement:
            reli = 2**instance.label - 1
        else:
            reli = 2**(1 if instance.label >= agreement_num else 0) - 1

        denom = log(i + 2,2) #denom = log2(i+2)
        result += reli / denom

    return result


def ndcg(dataset, agreement=True, agreement_num=1):
    result = dcg(dataset, agreement)

    idataset = deepcopy(dataset)
    for data in idataset:
        data.pred = data.label
    idcg = dcg(idataset, agreement, agreement_num=agreement_num)
    return result/idcg


def get_mrr(sentences, agreement=1):
    mrr = 0
    sorted_res = sorted(sentences, key=attrgetter("pred"), reverse=True)
    for i, res in enumerate(sorted_res):
        if res.label >= agreement:
            mrr += 1/(i+1)
    return mrr

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def get_all_metrics(sentences, agreement=1):
    """
    Prints list of results for a ranker.
    :param sentences:
    :return:
    """
    metrics = {'RR': [], 'AvgP': [], 'ROC': [], 'R_Prec':[],
               'nDCG_A': [], 'nDCG': [], 'Precision': [], 'Recall': [], 'Accuracy':[], 'F1':[],
               'Recall@10': [], 'Recall@100': [],'Recall@150': [], 'Recall@200': [], 'Recall@50':[],
               'PR@1': [], 'PR@3': [], 'PR@5': [],'PR@20': [], 'PR@10': [], 'PR@50': [], 'PR@100': [], 'PR@200': []}

    for sentence_set in sentences:
        sentence_set = (sorted(sentence_set, key=attrgetter("pred"), reverse=True)) # sort the sentence based on predictions in reverse order
        y_true = [1 if t.label >= agreement else 0 for t in sentence_set]
        y_pred = [s.pred for s in sentence_set]
        y_pred_label = [s.pred_label for s in sentence_set]

        metrics['AvgP'].append(average_precision(sentence_set, agreement=agreement))
        metrics['ROC'].append(roc_auc_score(y_true, y_pred))
        metrics['RR'].append(get_mrr(sentence_set, agreement=agreement))
        metrics['nDCG_A'].append(ndcg(deepcopy(sentence_set), agreement=True, agreement_num=agreement))
        metrics['nDCG'].append(ndcg(deepcopy(sentence_set), agreement=False, agreement_num=agreement))
        metrics['R_Prec'].append(r_precision(sentence_set, agreement=agreement))
        metrics['Precision'].append(precision_score(y_true, y_pred_label))
        metrics['Recall'].append(recall_score(y_true, y_pred_label))
        metrics['F1'].append(f1(y_true, y_pred_label))
        metrics['Accuracy'].append(accuracy(y_true, y_pred_label))

        for i in [1, 3, 5, 10, 20, 50, 100, 200]:
            metrics['PR@'+str(i)].append(precision_at_n(sentence_set, i, agreement=agreement))

        for i in [10, 50, 100, 150, 200]:
            metrics['Recall@'+str(i)].append(recall_at_n(sentence_set, i, agreement=agreement))

    for key, value in sorted(metrics.items(), key=itemgetter(0)):
        print("{0}\t\t {1:.4f}".format(key, mean(value)))

    print_for_table(metrics)


def print_for_table(metrics):
    order = ["Accuracy", "Precision", "Recall", "F1", "AvgP", "ROC",
             "RR", "PR@1", "PR@3", "PR@5", "PR@10", "PR@20", "PR@50",
             "PR@100", "PR@200", "Recall@10", "Recall@50", "Recall@100", "Recall@150", "Recall@200",
             "R_Prec", "nDCG"]
    labels = "\t".join([l for l in order])
    out = "\t".join(["{0:.4f}".format(mean(metrics[key])) for key in order])
    print(out)
    print(labels)


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

# main function to construct the pipeline of features and run a classifier
# if __name__ == '__main__':
#         results = []
#         for test_deb, test, train in get_for_crossvalidation():
#             results.append(run(train=train, test=test)) # running the neural network main function to train and predict
#             get_all_metrics(results, agreement=1)
#         get_all_metrics(results, agreement=1)