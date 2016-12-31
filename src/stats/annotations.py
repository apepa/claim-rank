from src.data.debates import read_all_debates
from src.utils.config import get_config

CONFIG = get_config()

output_file = open("../../reports/all_sources.txt", "w")

all_sentences = read_all_debates()

output_file.write("Overall sentences annotated: "+str(len(all_sentences))+"\n\n")

count_agreement = [0 for i in range(10)]
accumulated_agreement = [0 for i in range(10)]
for sent in all_sentences:
    count_agreement[sent.label] += 1

    for j in range(10):
        if sent.label >=j:
            accumulated_agreement[j] += 1

output_file.write("Agreement:\n")
for i in range(10):
    output_file.write(str(count_agreement[i]) +
                          " sentences with "+str(i)+" annotators agreed\n")

output_file.write("\nAccumulated Agreement:\n")
for i in range(1,10):
    output_file.write(str(accumulated_agreement[i])+" sentences with at least "+str(i)+" annotators agreed\n")

for i in range(9, -1, -1):
    output_file.write("\nSentences with "+str(i)+" agreed annotators:\n")
    for sent in all_sentences:
        if sent.label==i:
            output_file.write(sent.text+"\n")


