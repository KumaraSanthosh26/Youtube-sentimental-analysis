from google.oauth2.credentials import Credentials
import google.auth.transport.requests
import html
import google.auth
import os
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from googleapiclient.discovery import build
from flask import Blueprint, render_template, request
import nltk
nltk.download('movie_reviews')
nltk.download('punkt')

from flask import Blueprint

app_blueprint = Blueprint('app_blueprint', __name__)

DEVELOPER_KEY = 'AIzaSyBV-wkAyU-6XquTaG2zSCeKtqvFN0Iq5RA'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                developerKey=DEVELOPER_KEY)


@app_blueprint.route('/')
def index():
    return render_template("index.html")


@app_blueprint.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        video_id = request.form.get('video_id')

        # Fetch the comments from the video
        comments = []
        results = youtube.commentThreads().list(  # pylint: disable=maybe-no-member
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        ).execute()

        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # Train the naive Bayes classifier on labeled data - Unigrams
        train_data = [
            ("love it", "Positive"),
            ("great video", "Positive"),
            ("amazing work", "Positive"),
            ("thank you", "Positive"),
            ("awesome job", "Positive"),
            ("fantastic job", "Positive"),
            ("excellent video", "Positive"),
            ("impressive work", "Positive"),
            ("keep going", "Positive"),
            ("very informative", "Positive"),
            ("you're amazing", "Positive"),
            ("superb video", "Positive"),
            ("wonderful job", "Positive"),
            ("really enjoyed", "Positive"),
            ("really nice", "Positive"),
            ("so cool", "Positive"),
            ("pretty cool", "Positive"),
            ("so cute", "Positive"),
            ("amazing", "Positive"),
            ("love", "Positive"),
            ("great", "Positive"),
            ("perfect", "Positive"),
            ("fantastic", "Positive"),
            ("brilliant", "Positive"),
            ("awesome", "Positive"),
            ("incredible", "Positive"),
            ("impressive", "Positive"),
            ("cool", "Positive"),
            ("yay", "Positive"),
            ("legendary", "Positive"),
            ("fire", "Positive"),
            ("dope", "Positive"),
            ("chill", "Positive"),
            ("epic", "Positive"),
            ("adore", "Positive"),
            ("adorable", "Positive"),
            ("sweet", "Positive"),
            ("😍", "Positive"),
            ("😁", "Positive"),
            ("🤩", "Positive"),
            ("🥰", "Positive"),
            ("👍", "Positive"),
            ("❤️", "Positive"),
            ("🔥", "Positive"),
            ("not good", "Negative"),
            ("waste time", "Negative"),
            ("not funny", "Negative"),
            ("bad quality", "Negative"),
            ("terrible service", "Negative"),
            ("hate it", "Negative"),
            ("hate this", "Negative"),
            ("never buy", "Negative"),
            ("boring video", "Negative"),
            ("worst experience", "Negative"),
            ("not recommended", "Negative"),
            ("poor performance", "Negative"),
            ("this sucks", "Negative"),
            ("it sucks", "Negative"),
            ("never again", "Negative"),
            ("not impressed", "Negative"),
            ("stay away", "Negative"),
            ("not happy", "Negative"),
            ("won't recommend", "Negative"),
            ("rude staff", "Negative"),
            ("complete disaster", "Negative"),
            ("terrible food", "Negative"),
            ("not worth", "Negative"),
            ("boring content", "Negative"),
            ("hate", "Negative"),
            ("terrible", "Negative"),
            ("awful", "Negative"),
            ("bad", "Negative"),
            ("dislike", "Negative"),
            ("disgusting", "Negative"),
            ("disappointed", "Negative"),
            ("boring", "Negative"),
            ("annoying", "Negative"),
            ("pathetic", "Negative"),
            ("worst", "Negative"),
            ("lame", "Negative"),
            ("garbage", "Negative"),
            ("overrated", "Negative"),
            ("trash", "Negative"),
            ("abysmal", "Negative"),
            ("crappy", "Negative"),
            ("atrocious", "Negative"),
            ("useless", "Negative"),
            ("good video", "Neutral"),
            ("could have", "Neutral"),
            ("not bad", "Neutral"),
            ("nothing special", "Neutral"),
            ("not sure", "Neutral"),
            ("useful information", "Neutral"),
            ("kind of", "Neutral"),
            ("not really", "Neutral"),
            ("a bit", "Neutral"),
            ("somewhat interesting", "Neutral"),
            ("kinda cool", "Neutral"),
            ("fairly good", "Neutral"),
            ("okay", "Neutral"),
            ("okay content", "Neutral"),
        ]
        cl = NaiveBayesClassifier(train_data)

        # Perform sentiment analysis on the comments using the trained classifier
        positive_comments = 0
        negative_comments = 0
        neutral_comments = 0
        for comment in comments:
            blob = TextBlob(comment, classifier=cl)
            if blob.classify() == "Positive":
                positive_comments += 1
            elif blob.classify() == "Negative":
                negative_comments += 1
            else:
                neutral_comments += 1

        # Calculate the percentage of positive, negative, and neutral comments
        total_comments = len(comments)
        positive_percent = round((positive_comments / total_comments) * 100, 2)
        negative_percent = round((negative_comments / total_comments) * 100, 2)
        neutral_percent = round((neutral_comments / total_comments) * 100, 2)

        # Render the results page
        return render_template('result.html', positive=positive_percent, negative=negative_percent, neutral=neutral_percent)
