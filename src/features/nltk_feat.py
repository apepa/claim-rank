from src.features.features import Feature
from nltk.data import load
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk, tag


class POS(Feature):
    """Adds a vector of POS tag counts."""
    FEATS = ['pos']

    def __init__(self):
        tagdict = load('help/tagsets/upenn_tagset.pickle')
        tags_keys = tagdict.keys()
        self.tags = {}
        for i, tag in enumerate(tags_keys):
            self.tags[tag] = i

    def transform(self, X):
        for sent in X:
            tokenized = word_tokenize(sent.text)
            tag_vector = [0 for _ in range(len(self.tags))]
            pos_tags = pos_tag(tokenized)
            for word, tag in pos_tags:
                tag_vector[self.tags[tag]] += 1
            sent.features['pos'] = tag_vector
        return X


class NER(Feature):
    """Adds NEs count"""
    FEATS = ['ner_count_nltk']

    def transform(self, X):
        for sent in X:
            tokens = word_tokenize(sent.text)
            parse_tree = ne_chunk(tag.pos_tag(tokens), binary=True)  # POS tagging before chunking!

            named_entities = []

            for subtree in parse_tree.subtrees():
                if subtree.label() == 'NE':
                    named_entities.append(subtree)
            sent.features['ner_count_nltk'] = len(named_entities)
        return X
