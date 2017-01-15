from os.path import join
from os import listdir
from collections import defaultdict

resources_dict = {
    'tg': 'The Guardian',
    'npr': 'NPR',
    'pf': 'PolitiFact',
    'abc': 'ABC News',
    'nyt': 'The New York Times',
    'ct': 'Chicago Tribune',
    'wp': 'The Washington Post',
    'cnn': 'CNN',
    'fc': 'FactCheck.org'
}

files_dir = "../../data/transcripts_all_sources"
output_file = open("../../reports/sources_info.txt", "w")
for file in listdir(files_dir):
    ann_file = open(join(files_dir, file))

    headers = ann_file.readline().strip().lower().split("\t")

    sources = defaultdict(lambda: 0)
    all = 0
    ann_sents = 0
    for line in ann_file:
        columns = line.strip().split("\t")
        for i in range(3, 12):
            sources[headers[i]] += int(columns[i])
        all += int(columns[2])
        ann_sents += 1 if int(columns[2])>=1 else 0

    output_file.write("\n{} debate info:\n".format(file.split(".")[0][:-4]))
    for key, value in sources.items():
        output_file.write("{} annotations from {} \n".format(value, resources_dict[key]))
    output_file.write("{} annotations overall \n".format(all))
    output_file.write("{} annotated sentences overall\n".format(ann_sents))

    ann_file.close()

output_file.close()