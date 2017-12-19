from src.data.debates import read_all_debates, read_debates, Debate
from pyparsing import nestedExpr
from src.features.features import Feature
from src.utils.config import get_config
import subprocess

DISC_HOME = "/home/pepa/Downloads/Discourse_Parser_Dist/Discourse_Parser_Dist/"

def filter_parse_sent(parsed_sent):
    foundings = []
    for l in parsed_sent:
        if isinstance(l, str):
            pass
        elif l[0] == 'span':
            pass
        else:
            foundings.append(l)
    new_f = []
    if foundings[0][0] == 'DUMMY':
        foundings = foundings[0]
        for l1 in foundings:
            if l1[0] in ['rel2par', 'leaf'] or isinstance(l1, str):
                pass
            else:
                new_f.append(l1)
        return new_f
    for l in foundings:
        new_f.append(l)
    return new_f


def parse_list(text):
    text = text.replace("\'","")
    text = text.replace("\"", "")
    # print(text)
    res = nestedExpr('(', ')').parseString(text).asList()
    list = res[0]
    return list

def find_discourse(sent, chunk):
    if not chunk:
        return []
    new_sent = ",".join([str(s) for s in sent]).replace("\t","").replace(" ","")\
        # .replace("'","").replace("\"","").replace(",","").replace("\\","")
    # print(new_sent)
    for l in chunk:
        str_l = str(l).replace("\t","").replace(" ","")
        # print(str_l)
            # .replace("'","")\
            # .replace("\"","").replace(",","").replace("\\","").replace(")","]").replace("(","[")
        if new_sent in str_l:
            return find_discourse(sent, l) + l
    return []

def get_chunk_info(sent, chunk):
    found_discources = find_discourse(sent, chunk)
    relations = []
    discourses = []
    for l in found_discources:
        if l[0] == 'text':
            continue
        elif isinstance(l, str):
            discourses.append(l)
        elif l[0] == 'rel2par':
            # print("Relation:{}".format(l[1]))
            relations.append(l[1])
    return relations, discourses


def get_info(discourse_struct):
    rel = []
    it = []
    if all(isinstance(item, str) for item in discourse_struct) and discourse_struct[0] == 'rel2par' and not discourse_struct[1] == 'span':
        rel.append(discourse_struct[1])
    else:
        for el in discourse_struct:
            if el[0] not in ['leaf', 'span', 'text', 'rel2par']:
                if len(el[0]) < 2:
                    continue
                it.append(el[0])
            if not isinstance(el, str):
                rel_n, it_n = get_info(el)
                rel += rel_n
                it += it_n
    return rel, it

CONFIG = get_config()
chunks = {}
output_doc = open(CONFIG['chunk_parses'])
# print("Parsing chunks")
for line in output_doc:
    line = line.strip()
    if not line:
        continue
    cols = line.split("\t")
    # print(cols[0])
    chunks[int(cols[0])] = parse_list(cols[1])
output_sent = open(CONFIG['sentence_parses'])
sents = {}
# print("Parsing Sentences")
for line in output_sent:
    line = line.strip()
    if not line:
        continue
    cols = line.split("\t")
    # print(cols[0])
    sents[cols[0] + cols[1]] = parse_list(cols[2])
X = read_all_debates()
k = 0
sent2chunk = {}
curr_speaker = X[0].speaker
for i, sent in enumerate(X):
    if curr_speaker != sent.speaker:
        k += 1
        curr_speaker = sent.speaker
    sent2chunk["{}{}".format(sent.id, sent.debate.name)] = k


class DiscourseInfo(Feature):
    FEATS = ['discourse_it', 'discourse_rel', 'in_chunk_first_rel', 'in_chunk_last_rel', 'in_chunk_first_it',\
             'in_chunk_last_it', 'in_chunk_it', 'in_chunk_rel']

    RELS = ['DUMMY', 'Joint', 'Condition', 'Contrast', 'Joint', 'Attribution',
            'Enablement', 'Background', 'Elaboration', 'Cause', 'Explanation', 'Same-Unit',
            'Temporal', 'Manner-Means', 'span', 'Topic-Comment', 'TextualOrganization', 'Evaluation', 'Summary']
    ITS = ['DUMMY', 'Nucleus', 'Satellite']
    def prepare_it_vector(self, it):
        return [it.count(s) for s in self.ITS]

    def prepare_rel_vector(self, rel):
        return [rel.count(s) for s in self.RELS]

    def transform(self, X):
        print("Started Discourse")
        for i, sent in enumerate(X):
            # print(i)
            parse_sent = sents["{}{}".format(sent.id, sent.debate.name)]
            filtered_parse_sent = filter_parse_sent(parse_sent)
            chunk_id = sent2chunk["{}{}".format(sent.id, sent.debate.name)]
            parse_chunk = chunks[chunk_id]
            relations, discourses = get_chunk_info(filtered_parse_sent, parse_chunk)
            rel_sent, it_sent = get_info(parse_sent)
            discourses = [d for d in discourses if d in self.ITS]
            try:
                sent.features['in_chunk_first_rel'] = self.RELS.index(relations[0])
                sent.features['in_chunk_last_rel'] = self.RELS.index(relations[-1])
                sent.features['in_chunk_first_it'] = self.ITS.index(discourses[0])
                sent.features['in_chunk_last_it'] = self.ITS.index(discourses[-1])
            except Exception as e:
                sent.features['in_chunk_first_rel'] = -1
                sent.features['in_chunk_last_rel'] = -1
                sent.features['in_chunk_first_it'] = -1
                sent.features['in_chunk_last_it'] = -1

            sent.features['in_chunk_it'] = self.prepare_it_vector(discourses)
            sent.features['in_chunk_rel'] = self.prepare_rel_vector(relations)
            sent.features['discourse_it'] = self.prepare_it_vector(it_sent)
            sent.features['discourse_rel'] = self.prepare_rel_vector(rel_sent)
        print("Ended Discourse")
        return X

def get_parse(text, id):
    out = open(DISC_HOME + id, "w")
    out.write(text)
    out.close()
    run_cmd("python2 Discourse_Segmenter.py {}".format(id))
    run_cmd("python2 Discourse_Parser.py tmp.edu")
    out = open(DISC_HOME + "tmp_sen.dis")
    sent_parse = out.read()
    out.close()

    out1 = open(DISC_HOME + "tmp_doc.dis")
    doc_parse = out1.read()
    out1.close()

    # print(out_text)
    return sent_parse, doc_parse

import os
def run_cmd(cmd):
    d = dict(os.environ)
    d["WNSEARCHDIR"] = str("/home/pepa/nltk_data/corpora")
    d["WNHOME"] = str("/home/pepa/nltk_data/corpora")
    d["wnHomeUnix"] = str("/home/pepa/nltk_data/corpora/wordnet")
    d["wnPrefixUnix"] = str("/home/pepa/nltk_data/corpora/wordn")
    d["PERL5LIB"] = "/home/pepa/nltk_data/corpora"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, env=d, cwd=DISC_HOME)
    output, error = process.communicate()
    # print(output)
    # print(error)

def generate_all():
    X = read_all_debates()
    k = 0
    CHUNKS = [""]
    curr_speaker = X[0].speaker
    for i, sent in enumerate(X):
        if curr_speaker != sent.speaker:
            k += 1
            CHUNKS.append("")
            curr_speaker = sent.speaker
        sent.features['mod_chunk'] = k
        # if 'audit is over' in sent.text:
        #     print(k)
        #     print(CHUNKS[k])
        CHUNKS[k] = CHUNKS[k] + " " + sent.text
    CHUNKS.append("")
    # print(k)
    output_doc = open("/home/pepa/PycharmProjects/claim-rank/data/discourse/chunk_parses_new_680.txt", "w")
    output_sent = open("/home/pepa/PycharmProjects/claim-rank/data/discourse/sent_parses_new_680.txt", "w")
    j = 0

    start = 680
    end = 750
    for i, chunk in enumerate(CHUNKS):
        if i < start or i > end:
            continue
        # print(i)
        parse_sent, parse_doc = get_parse(chunk, "{}".format(i))
        parse_doc = parse_doc.replace("\n", "")
        output_doc.write("{}\t{}\n".format(i, parse_doc))

        for s, p_s in enumerate(parse_sent.split(")\n(")):
            if s == 0:
                p_s = p_s + ")"
            elif s == len(parse_sent.split(")\n("))-1:
                p_s = "("+p_s
            else:
                p_s = "(" + p_s + ")"

            output_sent.write("{}\t{}\t{}\n".format(X[j].id, X[j].debate.name, p_s.replace("\n", "")))
            j += 1

        # info = get_info(parse_list(parse))
        # output.write("{}\t{}\t{}\t{}\n".format(sent.debate.name, sent.id, ",".join(info[0]), ",".join(info[1])))
        # output.close()

# generate_all()
sent = "( Root (span 4 5)\
  ( Nucleus (leaf 4) (rel2par span) (text _!It 's about putting money_!) )\
  ( Satellite (leaf 5) (rel2par Elaboration) (text _!-- more money into the pockets of American workers ._!) )\
)"

chunk = "( Root (span 1 6)\
  ( Nucleus (span 1 2) (rel2par span)\
    ( Nucleus (leaf 1) (rel2par span) (text _!Secretary Clinton ,_!) )\
    ( Satellite (leaf 2) (rel2par Elaboration) (text _!thank you ._!) )\
  )\
  ( Satellite (span 3 6) (rel2par Elaboration)\
    ( Nucleus (leaf 3) (rel2par span) (text _!Mr. Trump , the same question to you ._!) )\
    ( Satellite (span 4 6) (rel2par Elaboration)\
      ( Nucleus (span 4 5) (rel2par span)\
        ( Nucleus (leaf 4) (rel2par span) (text _!It 's about putting money_!) )\
        ( Satellite (leaf 5) (rel2par Elaboration) (text _!-- more money into the pockets of American workers ._!) )\
      )\
      ( Satellite (leaf 6) (rel2par Elaboration) (text _!You have up to two minutes ._!) )\
    )\
  )\
)"

# parse_chunk = parse_list(chunk)
# parse_sent = parse_list(sent)

# print(parse_sent)


# filtered_parse_sent = filter_parse_sent(parse_sent)
# print(filtered_parse_sent)
# print(parse_chunk)



# res = find_discourse(filtered_parse_sent, parse_chunk)
# print(len(res))

# print(get_chunk_info(filtered_parse_sent, parse_chunk))
# print(get_info(parse_sent))