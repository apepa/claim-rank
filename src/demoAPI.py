import sys
#import re
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
from src.data.models import *
from src.data.main import *
from flask import Flask,session, render_template,request,redirect, url_for,escape
app = Flask(__name__)
app.secret_key = 'IsraaQasim89'

# if the button "showResults" is clicked , redirect the user to the results page carrying the textfield value in the request object
@app.route("/showResults", methods=['POST'])
def showResults():
    text = request.form['transcript']
    obj = mainclass()

######################################################################################################################################################################################
########################################################################   M A I N      C A L L  #####################################################################################
######################################################################################################################################################################################
# send the text, if you want to train the model set second param to True, if you want to predict scores for the entered text set thirs param to True
# method = 'svm' or 'nn' based on the method you want to use for learning  (Support Vector Machines , Neural Netwok)

    sentenceVec = obj.mainproc(text, True, True, 'svm' )

#######################################################################################################################################################################################
#######################################################################################################################################################################################
#######################################################################################################################################################################################

    sentenceCount = len(sentenceVec)
    sentencelist = []
    scoreList = []
    i = 0
    # converting the list of objects to a built in python parallel arrays so they can be JSONfied to be stored in the session (session in python is a JSON objects)
    for sent in sentenceVec:
        sent.pred = round(sent.pred, 2)
        sentencelist.append(sent.text)
        scoreList.append(sent.pred)
        i+=1

    session['sentenceList'] = sentencelist
    session['scoreList'] = scoreList
    return render_template('results.html', transSentenceList=sentenceVec, sentenceCount=sentenceCount)


@app.route("/sortResults", methods=['POST'])
def sortResults():
    sentenceList = session['sentenceList']
    scoreList = session['scoreList']
    sentenceVec = []
    i = 0
    for sent in sentenceList:
        sentence = demoSentence(str(i), sent)
        sentence.pred = scoreList[i]
        sentenceVec.append(sentence)
        i += 1

    sentenceVec.sort(key=lambda sent: sent.pred, reverse=True)
    sentenceCount = len(sentenceVec)
    return render_template('results.html', transSentenceList=sentenceVec, sentenceCount=sentenceCount)


# direct the user to the main page when the app starts
@app.route("/")
def main():
	return render_template('index.html')



if __name__=="__main__" :
    app.run()





