from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from os.path import join

from src.data.debates import Debate
from src.data.debates import read_debates
from src.utils.config import get_config
from src.utils.timing import timing

CONFIG = get_config()

class ClaimBusterScraper(object):
    """
    Used for scraping the score number of the ClaimBuster Engine for a given sentence.
    Selenium is used for scraping.

    >>>cb = ClaimBusterScraper()
    >>>cb.get_score("That is wrong!")
    """
    CB_URL = "http://idir-server2.uta.edu/claimbuster/demo"

    def initialize(self):
        self.driver = webdriver.PhantomJS('../../phantomjs/bin/phantomjs')
        self.driver.get(self.CB_URL)

    def __init__(self):
        self.initialize()

    def get_score(self, text):
        """
        :param text: the sentence (chunk of text) to get the score for
        :return: float number, representing the score, given by CB for the input text
        """
        elem = self.driver.find_element_by_id("buttonEdit")
        elem.click()

        elem = self.driver.find_element_by_id("divTranscript")
        elem.click()
        elem.send_keys(text)

        button = self.driver.find_element_by_id("buttonSubmit")
        button.click()
        count = 0
        score = -1
        while score == -1:
            count += 1
            if count > 20:
                count = 0
                self.initialize()
                return self.get_score(text)
            try:
                score = self.driver.find_element_by_xpath('// *[ @ id = "divTranscript"] / span').text
            except NoSuchElementException:
                pass

        return float(score)


class ClaimBusterAnnotate(object):
    """
    Used for serializing results from scraping the CB Engine.

    >>>cba = ClaimBusterAnnotate()
    >>>cba.annotate(Debate.FIRST)
    """
    SEP = "\t"
    NL = "\n"
    FILE_EXT = "_cb.tsv"

    def __init__(self):
        self.cb_scraper = ClaimBusterScraper()

    @timing
    def annotate(self, debate):
        """
        :param debate: debate (Debate enum) to get scores of sentences for.
        """
        debate_sentences = read_debates(debate)

        cb_output_name = join(CONFIG['tr_cb_anns'], CONFIG[debate.name] + self.FILE_EXT)

        cb_output = open(cb_output_name, 'w')
        cb_output.write("ID"+self.SEP+"Speaker"+self.SEP+"CB"+self.SEP+"Text"+self.NL)
        for sentence in debate_sentences:
            cb_score = self.cb_scraper.get_score(sentence.text)

            new_line = sentence.id+self.SEP + \
                sentence.speaker+self.SEP + \
                str(cb_score)+self.SEP + \
                sentence.text+self.NL
            print(new_line)

            cb_output.write(new_line)

        cb_output.close()