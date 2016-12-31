import numpy as np
from sklearn.metrics import precision_recall_curve,average_precision_score
from src.data.debates import read_all_debates
from src.utils.config import *
import matplotlib.pyplot as plt
from itertools import cycle

CONFIG = get_config()
COLORS = cycle(['mediumpurple', 'royalblue', 'lightblue', 'indianred', 'm', 'lightpink', 'hotpink'])


def pr_curve(agreement=1):
    """
    Saves Precision-Recall Curves at various threshold of the Claim Buster's scores.
    :param agreement: the agreement of our annotators to use when selecting positives.
    :return:
    """
    auc, precisions, recalls, thresholds = get_pr(agreement)

    plt.clf()
    for i, color in zip(range(len(precisions)), COLORS):
        plt.plot(recalls[i], precisions[i], color=color, lw=2,
                 label='PR at threshold {:.2f}, AUC={:.2f}'
                       .format(thresholds[i], auc[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve of CB  \n(Our positives are with >={} agreement)'.format(agreement))
    plt.legend(loc="upper right")
    plt.savefig(CONFIG['pr_curves_dir']+"pr_"+str(agreement)+"_agreement.png")


def get_pr(agreement):
    precisions = []
    recalls = []
    auc = []
    thresholds = []
    y_prob = [sent.label for sent in read_all_debates(source='cb')]
    y_score = [1 if sent.label >= agreement else 0 for sent in read_all_debates()]
    step = 0.1
    for i in np.arange(0.4, 1, step):
        thresholds.append(float(i))
        y_test = [1 if x > i else 0 for x in y_prob]

        precision, recall, _ = precision_recall_curve(y_test,
                                                      y_score)
        precisions.append(precision)
        recalls.append(recall)
        average_precision = average_precision_score(y_test, y_score)
        auc.append(float(average_precision))
    return auc, precisions, recalls, thresholds


for i in range(1, 7):
    pr_curve(agreement=i)
