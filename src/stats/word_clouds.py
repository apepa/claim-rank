from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

from src.data.debates import read_all_debates
from src.utils.config import get_config

CONFIG = get_config()
WC = WordCloud(max_font_size=40, background_color='white')


def get_tokens(sentences):
    text = " ".join([sent.text.lower() for sent in sentences])
    stop = list(stopwords.words('english'))
    tokens = [token for token in word_tokenize(text) if token not in stop]
    return " ".join(tokens)


def save_wordcloud(wordcloud, file_name):
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(CONFIG['word_clouds_dir'] + file_name)


all_sents = read_all_debates()
save_wordcloud(WC.generate(get_tokens(all_sents)), "all_sents.png")

negative_sents = [sent for sent in all_sents if sent.label == 0]
save_wordcloud(WC.generate(get_tokens(negative_sents)), "neg_sents.png")

positive_sents = [sent for sent in all_sents if sent.label > 0]
save_wordcloud(WC.generate(get_tokens(positive_sents)), "pos_sents.png")

for i in range(1,9):
    pos_gt_i = [sent for sent in all_sents if sent.label > i]
    save_wordcloud(WC.generate(get_tokens(pos_gt_i)), "pos_gt_"+str(i)+"_sents.png")

all_cb = read_all_debates(source='cb')
for i in np.arange(0.5,1.1,0.1):
    pos_gt_i = [sent for sent in all_sents if sent.label > i]
    save_wordcloud(WC.generate(get_tokens(pos_gt_i)), "pos_gt_" + str(i) + "_cb_sents.png")
