from os.path import join
from os import listdir
from collections import defaultdict

files_dir = "../../data/transcripts_all_sources"
output_file = open("../../reports/claim_chunks.tsv", "w")
i = 0
for file in listdir(files_dir):
    ann_file = open(join(files_dir, file))

    output_file.write("Debate\t"+ann_file.readline())
    chunk = []
    for line in ann_file:
        columns = line.strip().split("\t")
        ann_curr = int(columns[2]) >= 1

        columns = [file.split(".")[0][:-4]] + columns
        if ann_curr:
            # sentence is annotated
            chunk.append("\t".join(columns))
        else:
            # sentence is not annotated
            if len(chunk)<=1:
                chunk = []
            else:
                output_file.write("\nChunk #{}:\n".format(i))
                i+=1
                for sent in chunk:
                    output_file.write(sent+"\n")
                chunk = []

output_file.close()
