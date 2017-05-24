from src.utils.dicts import get_stopwords
from src.features.features import Feature
from nltk.tokenize import word_tokenize
from src.utils.config import get_config

import json
import datetime

CONFIG = get_config()
CLAIMS_FILE = CONFIG['pf_claims']


def text2date(text):
    date_numbers = text.split("T")[0].split("-")
    date = datetime.datetime(year=int(date_numbers[0]), month=int(date_numbers[1]), day=int(date_numbers[2]))
    return date


class PolitiFactSearch(Feature):
    """
    Adds score for similarity to a nearest existing check-worthy claim.
    The claims are from Politi-Fact's API.
    """
    FEATS = ["pf_match_score", "pf_match_with_person"]

    def __init__(self):

        self.stopwords = get_stopwords()
        claims_file = open(CLAIMS_FILE).read()
        all_claims = json.loads(claims_file)
        self.claims = []
        for claim in all_claims:
            claim_text = claim['statement'].replace("<p>", "").replace("</p>", "").replace("\"", "").replace("&quot;",                                                                                            "")
            claim_tokens = [w for w in word_tokenize(claim_text) if w not in self.stopwords]
            claim_date = text2date(claim['ruling_date'])
            speaker = claim['speaker']['last_name']
            self.claims.append((claim_tokens, claim_date, speaker))

    def transform(self, X):
        for sent in X:
            sent_clean_tokens = [word for word in sent.tokens if word not in self.stopwords]
            sim_to_nearest_claim = 0
            sim_with_person = 0
            for pf_claim in self.claims:
                # how many tokens overlap in both claims
                overlapping_tokens = len(set(sent_clean_tokens).intersection(pf_claim[0]))
                if overlapping_tokens > 0:
                    if text2date(sent.date) > pf_claim[1]:
                        # only take this score if the claim was said earlier! otherwise-overfitting
                        sim_to_nearest_claim = max(sim_to_nearest_claim, overlapping_tokens)
                        sim_with_person = max(sim_with_person, overlapping_tokens*(1 if pf_claim[2] == sent.speaker else 0))
                sent.features["pf_match_score"] = sim_to_nearest_claim
                sent.features["pf_match_with_person"] = sim_with_person
        return X


class TrainSearch(Feature):
    """
       Adds score for similarity to a nearest existing check-worthy claim.
       The claims are from the train set.
    """
    FEATS = ["match_score", "match_with_person"]

    def __init__(self, train=None):
        self.train = train
        self.stopwords = get_stopwords()

        for sent in self.train:
            sent.clean_tokens = [w for w in sent.tokens if w not in self.stopwords]

    def transform(self, X):
        for sent in X:
            clean_tokens = [w for w in sent.tokens if w not in self.stopwords]
            sim_to_nearest_claim = 0
            sim_with_person = 0
            for train_sent in self.train:
                # how many tokens overlap in both claims
                overlapping_tokens = len(set(clean_tokens).intersection(train_sent.clean_tokens))
                if overlapping_tokens > 0:
                    # only take similarity to claim-worthy ones!
                    claim = 1 if train_sent.label > 0 else 0
                    sim_to_nearest_claim = max(claim * overlapping_tokens, sim_to_nearest_claim)
                    sim_with_person = max(claim * overlapping_tokens
                                            * (1 if train_sent.speaker == sent.speaker else 0), sim_with_person)

            sent.features['match_score'] = sim_to_nearest_claim
            sent.features['match_with_person'] = sim_with_person
        return X

