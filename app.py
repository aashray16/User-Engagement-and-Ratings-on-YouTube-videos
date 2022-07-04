from flask import Flask, render_template, request
from apiclient.discovery import build
import pandas as pd
import numpy as np
import requests
import json
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
import regex as re
from autocorrect import Speller
import nltk
import csv

Api_Key = "AIzaSyCweOZ94C5VuKZeDrQTwf3KOXBPVV4fh3w"
youtube = build("youtube", "v3", developerKey=Api_Key)

app = Flask(__name__)

commentTfidf = pickle.load(open("Commenttifdf.pkl", "rb" ))
commentModel = pickle.load(open("CommentModel.pkl", 'rb'))
titleModel = keras.models.load_model("titleFinal.h5")
with open('titletokenizer.pickle', 'rb') as handle:
    titleTokenizer = pickle.load(handle)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predictratings", methods = ["POST"])
def predictratings():
    ID = request.form["videoID"]
    video = youtube.videos().list(part='snippet,statistics',id=ID).execute()
    channel_id = video['items'][0]['snippet']['channelId']
    channel_info = youtube.channels().list(id=channel_id,part="statistics").execute()
    res = requests.get("https://returnyoutubedislikeapi.com/votes?videoId=" + ID)
    response = json.loads(res.text)
    Title = video['items'][0]['snippet']['title']
    Views = video['items'][0]['statistics']['viewCount']
    Likes = video['items'][0]['statistics']['likeCount']
    Dislikes = response["dislikes"]
    CommentCount = video['items'][0]['statistics']['commentCount']
    SubCount = channel_info['items'][0]['statistics']['subscriberCount']
    List = []
    data = youtube.commentThreads().list(part="snippet", videoId=ID, maxResults="100", textFormat="plainText").execute()
    for i in data["items"]:
        comment = i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        List.append([comment])

    while ("nextPageToken" in data):
        data = youtube.commentThreads().list(part="snippet", videoId=ID, pageToken=data["nextPageToken"], maxResults="100", textFormat="plainText").execute()
        for i in data["items"]:
            comment = i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            List.append([comment])
        
    df = pd.DataFrame({"Comment": [i[0] for i in List]})
    df.to_csv(ID+".csv", index=False, header=False)
    df['realComment'] = df['Comment']
    spell = Speller(lang='en')
    def typo_corrector(text):
        return spell(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    def lemmatize_text(text):
        return lemmatizer.lemmatize(text)
    df['Comment'] = df['Comment'].str.lower()
    df['Comment'] = df['Comment'].str.replace('http\S+|www.\S+', '', case=False)          #url
    df['Comment'] = df['Comment'].str.replace('\n',' ', regex=True)                       #lines
    df['Comment'] = df['Comment'].str.replace('[^\w\s]',' ')                              #punctuations
    df['Comment'] = df['Comment'].str.replace('\d','', regex=True)                        #int
    df['Comment'] = df['Comment'].str.replace('[^\w\s#@/:%.,_-]', ' ', flags=re.UNICODE)  #emoji
    df['Comment'] = df['Comment'].apply(typo_corrector)                                   #autocorrect
    df['Comment'] = df['Comment'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    cmtFeatures = commentTfidf.transform(df['Comment'])
    df['Type'] = commentModel.predict(cmtFeatures)
    df = df.sort_values(by=['Type'])
    df = df.drop(['Comment'], axis=1)
    df = df.reset_index(drop=True)
    df.to_csv('CommentType'+ID+'.csv')
    txt = titleTokenizer.texts_to_sequences([Title])
    txt = sequence.pad_sequences(txt, maxlen=200)
    titlePred = (titleModel.predict(txt))
    titleRate = titlePred[0][0] * 10
    titleRate = round(titleRate, 1)
    CommentsToViews = round((int(CommentCount)/(int(Views)/200))*10,1)  # out of 10
    if CommentsToViews > 10 : CommentsToViews = 10
    LikesToViews = round((int(Likes)/(int(Views)/25))*10,1) # out of 10
    if LikesToViews > 10 : LikesToViews = 10
    ViewsToSubs = round((int(Views)/(int(SubCount)/7.143))*10,1)
    if ViewsToSubs > 10 : ViewsToSubs = 10
    LikesToDislikes = round(int(Likes)/(int(Likes)+int(Dislikes))*10,1)
    negComments = 0
    if ('negative' in df['Type'].unique()):
        negComments = df['Type'].value_counts().negative
    CommentRatio = round((len(df.index)-negComments)/len(df.index)*10, 1)
    VideoRating = round((CommentRatio + titleRate + CommentsToViews*0.5 + LikesToViews*0.5 + LikesToDislikes + ViewsToSubs)/5, 1) 
    if titleRate < 8.0:
        return render_template('index.html', rating = "Ratings are  {} / 10".format(VideoRating))

@app.route('/titleTest')
def titleTest():
    return render_template("titleTest.html")

@app.route('/predicttitle', methods = ["POST"])
def predicttitle():
    text = request.form["titleTxt"]
    txt = titleTokenizer.texts_to_sequences([text])
    txt = sequence.pad_sequences(txt, maxlen=200)
    titleRate = (titleModel.predict(txt))
    titleRate = round(titleRate[0][0]*10,1)
    if titleRate <= 8.0:
        return render_template('titleTest.html', titlerating = "Ratings are  {} / 10".format(titleRate), ratetext = "You should try something different.") 
    elif titleRate > 8.0 and titleRate<=9.0:
        return render_template('titleTest.html', titlerating = "Ratings are  {} / 10".format(titleRate), ratetext = "Its Good!! But you should try something catchier.") 
    else:
        return render_template('titleTest.html', titlerating = "Ratings are  {} / 10".format(titleRate), ratetext = "Thats a perfect one!!") 



@app.route('/viewComments')
def viewComments():
    return render_template("viewComments.html")

@app.route('/predictComments', methods = ["POST"])
def predictComments():
    ID = request.form["videoIDC"]
    example = pd.read_csv('CommentType'+ID+'.csv')
    if ('other' in example['Type'].unique()):
        example = example[example.Type != 'other']
    example = example.reset_index(drop=True)
    return render_template('viewComments.html', data = example.to_html(header=False, index=False))
    

if __name__ == "__main__":
    app.run(debug=True)
