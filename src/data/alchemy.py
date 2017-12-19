from watson_developer_cloud import AlchemyLanguageV1

from data.debates import read_debates, Debate
from src.utils.config import get_config
from time import sleep
CONFIG = get_config()


def serialize_sentiment(sentences):
    """
    Serialize sentences sentiment.
    :param sentences: sentences to get sentiment for
    :return:
    """
    sentiment_output_file = open("sentiment.tsv", "a")
    sentiment_output_file.write("Id\tDebate\tText\tType\tScore\n")
    alchemy_language = AlchemyLanguageV1(api_key=CONFIG['API'])

    for i, sent in enumerate(sentences):
        try:
            result = alchemy_language.sentiment(text=sent.text, language='english')['docSentiment']
        except Exception:  # bad connection, etc., retry several times
            print(i)
            sleep(10)
            alchemy_language = AlchemyLanguageV1(api_key=CONFIG['API_KEY_ALCHEMY'])
            result = alchemy_language.sentiment(text=sent.text, language='english')['docSentiment']

        print(result)
        print(sent.text)
        sentiment_output_file.write("{}\t{}\t{}\t{}\t{}\n".format(sent.id, sent.debate.name, sent.text,
                                                                  result.get('score', 0.0), result['type']))
    sentiment_output_file.close()


def get_ner_types():
    types_file = open("../../data/dicts/alchemy_types.txt")
    types = []
    for line in types_file:
        types.append(line.strip())

    subtypes_file = open("../../data/dicts/alchemy_subtypes.txt")
    subtypes = []
    for line in subtypes_file:
        subtypes.append(line.strip())
    return types, subtypes


def serialize_ner():
    alchemy_language = AlchemyLanguageV1(api_key=CONFIG['API'])

    types, subtypes = get_ner_types()

    results = open("../../data/dicts/ner.tsv", "a")
    results.write("ID\tDebate")
    for key in types:
        results.write("\t"+str(key))
    for key in subtypes:
        results.write("\t"+str(key))
    results.write("\n")

    sentences = read_debates(Debate.CLINTON_S)[78:]

    for sentence in sentences:
        try:
            ner = alchemy_language.entities(text=sentence.text)['entities']
        except Exception:
            print(sentence.text)
            sleep(10)
            alchemy_language = AlchemyLanguageV1(api_key=CONFIG['API'])
            ner = alchemy_language.entities(text=sentence.text, language='en')['entities']

        s_types = [0 for _ in range(len(types))]
        s_sub_types = [0 for _ in range(len(subtypes))]
        for ne in ner:
            type = ne.get('type', 0)
            if type!=0:
                s_types[types.index(type)] += 1
            subtype = ne.get('disambiguated', 0)
            if subtype != 0:
                subtype = subtype.get('subType', 0)
            if subtype != 0:
                for s in subtype:
                    s_sub_types[subtypes.index(s)]+=1

        results.write(sentence.id+"\t"+sentence.debate.name)
        for value in s_types:
            results.write("\t{}".format(value))
        for value in s_sub_types:
            results.write("\t{}".format(value))
        results.write("\n")
    results.close()


def ner_non_zeroes():
    # remove all ners that are not met in out data
    ner_file = open("../../data/dicts/all_ner.tsv")

    types, subtypes = get_ner_types()
    types += subtypes

    types_sum = [0 for _ in range(len(types))]

    heading = ner_file.readline()
    headings = heading.strip().split("\t")
    for line in ner_file:
        line = line.strip()
        columns = line.split("\t")
        for i, value in enumerate(columns):
            if i > 1 and int(value) != 0 and headings[i] in types:
                types_sum[i - 2] += int(value)

    ner_file.close()
    # print(types_sum)
    for i in range(len(types)):
        print("{},{}".format(types[i], types_sum[i]))

    # print(only_non_zero)
    ner_file = open("../../data/dicts/all_ner.tsv")
    new_ner_file = open("../../data/dicts/ner_no_zeroes.tsv", "w")

    headings = ner_file.readline().strip().split("\t")

    for i, head in enumerate(headings):
        if i == 0:
            new_ner_file.write(head)
        elif i == 2:
            new_ner_file.write("\t"+head)
        else:
            if types_sum[i-2]>0:
                new_ner_file.write("\t" + head)
    new_ner_file.write("\n")

    for line in ner_file:
        line = line.strip()
        columns = line.split("\t")
        for i, value in enumerate(columns):
            if i == 0:
                new_ner_file.write(value)
            elif i == 1:
                new_ner_file.write("\t"+value)
            else:
                if types_sum[i-2] > 0:
                    new_ner_file.write("\t" + value)
        new_ner_file.write("\n")

if __name__ == "__main__":
    # ner_non_zeroes()
    # sentences = read_debates(Debate.TRUMP_I)
    # serialize_sentiment(sentences)

    serialize_ner()

