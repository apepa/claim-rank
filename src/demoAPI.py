import sys
#import re
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
import codecs
from src.data.models import *
from src.data.main import *
from flask import Flask, render_template,request,redirect, url_for
app = Flask(__name__)

# if the button "showResults" is clicked , redirect the user to the results page carrying the textfield value in the request object
@app.route("/showResults", methods=['POST'])
def showResults():
    text = request.form['transcript']
    obj = mainclass()


    #                               M A I N        C A L L
    #########################################################################################################################################################
    # send the text, if you want to train the model set second param to True, if you want to predict scores for the entered text set thirs param to True    #
    sentenceVec = obj.mainproc(text, False, True)                                                                                                             #
    #########################################################################################################################################################


    sentenceCount = len(sentenceVec)
    # extract two separate arrays from
    return render_template('results.html', transSentenceList = sentenceVec, sentenceCount= sentenceCount)



# direct the user to the main page when the app starts
@app.route("/")
def main():
	return render_template('index.html')

if __name__ == "__main__":
	app.run()




