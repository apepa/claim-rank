from src.data.debates import get_for_crossvalidation, read_debates, read_cb_scores
from src.models.svm_rank import run_svm_rank
from os.path import join
from src.data.svm_converter import read_svm_pred
from src.stats.rank_metrics import get_metrics_for_plot, get_all_metrics
from src.utils.config import *
import matplotlib.pyplot as plt
from itertools import cycle

CONFIG = get_config()
COLORS = cycle(['mediumpurple', 'royalblue', 'lightblue', 'indianred', 'm', 'lightpink', 'hotpink'])


def pr_one_curve(agreement, ranks, color='navy', rank_name='CB'):
    """ Plots one PR curve. """
    avg_precisions, precision, recall, thresholds = get_metrics_for_plot(agreement, ranks)

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

def plot_multiple_predictors(ranks, names):
    """
    :param ranks: list of ranks, given by several ranking engines
    :param names: list of the names of the ranking enignes
    :return:

    """
    for i in range(1, 7):
        for rank, color, name in zip(ranks, COLORS, names):
            pr_one_curve(ranks=rank, agreement=i, color=color, rank_name=name)
        plt.clf()

if __name__ == '__main__':
    cb = []
    rank_svm = []
    for test_debate, test, train in get_for_crossvalidation():
        cb.append(read_cb_scores(test_debate))
        rank_svm.append(read_svm_pred(test, join(CONFIG['svm_rank'], test_debate.name)))

    plot_multiple_predictors([cb, rank_svm], ['CB', "SVM_RANK"])
