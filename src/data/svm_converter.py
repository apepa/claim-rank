import operator
import os
from sklearn.preprocessing.data import MinMaxScaler
ATTR_SEP = " "
INST_SEP = "\n"


def save_for_svm_rank(data, labels, output_file):
    if len(labels) == 0:
        step = -1
        labels = [i for i in range(len(data), 0, step)]

    output = open(output_file, "w")
    output.write("#query 1\n")
    for instance, clazz in zip(data, labels):
        output.write(str(clazz)+ATTR_SEP+"qid:1"+str(ATTR_SEP))
        for i, attr in enumerate(instance):
            output.write("{0}:{1} ".format(i+1, attr))
        output.write(INST_SEP)
    output.close()


def read_svm_pred(test_sent, input_file):
    input = open(input_file)
    ranks = []
    for line in input:
        ranks.append(float(line.strip()))

    ranks = MinMaxScaler().fit_transform(ranks)
    for i, sent in enumerate(test_sent):
        test_sent[i].pred = ranks[i]
        test_sent[i].pred_label = 1 if ranks[i] >= 0.5 else 0
    return test_sent
