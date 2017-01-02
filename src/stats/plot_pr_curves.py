from sklearn.metrics import average_precision_score, precision_score, recall_score
from src.data.debates import read_all_debates, read_debates, Debate, read_cb_scores
from src.models.svm_rank import run_svm_rank_crossval, run_svm_rank
from src.utils.config import *
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

CONFIG = get_config()
COLORS = cycle(['mediumpurple', 'royalblue', 'lightblue', 'indianred', 'm', 'lightpink', 'hotpink'])


def pr_one_curve(agreement, ranks, sents, color='navy', rank_name='CB'):
    """ Plots one PR curve. """
    avg_precisions, precision, recall, thresholds = get_metrics(agreement, ranks, sents)

    plt.subplots_adjust(bottom=0.1)
    plt.plot(recall, precision, color=color, label=rank_name)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall of Claim Buster and SVM-Rank at each threshold\n (from 0 to 1 with step 0.05)')
    plt.legend(loc="lower right")

    i = 0
    for threshold, x, y, avgp in zip(thresholds, recall, precision, avg_precisions):
        if i%5==0:
            plt.annotate(
                "th:{:.2f}".format(threshold),
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        i += 1

    plt.savefig(CONFIG['pr_curves_dir']+"pr_cb_{}_agreement.png".format(agreement))


def get_metrics(agreement, ranks, sents):
    """
    Calculates the precision, recalls and avg. precisions at various threshold levels.
    :param agreement: level of agreement for a cliam to be counted as positive
    :param ranks: ranks given by a classifier/ranker
    :param sents: the sentences dataset
    :return: average precisions, precisions, recalls, thresholds
    """
    precision = []
    recall = []
    av_p = []

    # get for positives claims with agreement above 'agreement'
    y_score = [1 if sent.label >= agreement else 0 for sent in sents]

    step = 0.01
    thresholds = [i for i in np.arange(0, 1 + step, step)]

    for threshold in thresholds:
        # get for positive predictions those with rank above threshold
        y_test = [1 if x > threshold else 0 for x in ranks]
        precision.append(precision_score(y_true=y_score, y_pred=y_test))
        recall.append(recall_score(y_true=y_score, y_pred=y_test))
        av_p.append(average_precision_score(y_true=y_score, y_score=y_test))

    return av_p, precision, recall, thresholds


def plot(sents, ranks, names):
    """
    :param sents: sentences to be used for plotting
    :param ranks: list of ranks, given by several ranking engines
    :param names: list of the names of the ranking enignes
    :return:

    >>>all_sents = read_all_debates()
    >>>cb_ranks = [sent.label for sent in read_all_debates(source='cb')]
    >>>rank_svm = [x.rank for x in run_svm_rank_crossval(C=1.7)]
    >>>plot(all_sents, [cb_ranks, rank_svm], ['CB', "SVM_RANK"])
    """
    for i in range(1, 7):
        for rank, color, name in zip(ranks, COLORS, names):
            pr_one_curve(sents=sents, ranks=rank, agreement=i, color=color, rank_name=name)
        plt.clf()
