import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import numpy as np
import pandas as pd
import tweepy as tw
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from dotenv import load_dotenv

from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import re
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from transformers import pipeline

#loading model and tokenizer for movies
model_movies = load_model('model_movies.h5')
with open('tokenizer_movies.pickle', 'rb') as tokenizer_movies:
    tokenizer_movies = pickle.load(tokenizer_movies)
#loading model and tokenizer for twitter analysis
model_twitter = load_model('adrish_model.h5')
with open('tokenizer_twitter.pickle', 'rb') as tokenizer_twitter:
    tokenizer_twitter = pickle.load(tokenizer_twitter)

def preprocess_movies(review):
    
    # convert to lowercase
    text = review.lower()
    #remove html text
    soup = BeautifulSoup(text, "html.parser")
    text= soup.get_text()
    # remove non letters special characters
    text = re.sub(r'[^a-zA-z0-9\s]', " ", text)
    #removing between square brackets
    text=re.sub('\[[^]]*\]', '', text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # stemming
    words = [PorterStemmer().stem(w) for w in words]
    #return list
    return words

def predict_class_movies(text):
    
    sentiment_classes = ['Negative', 'Positive']
    max_len=600

    #text=preprocess(text)
    xt = tokenizer_movies.texts_to_sequences(text)

    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model_movies.predict(xt).argmax(axis=1)

    return sentiment_classes[yt[0]]


def predict_class_twitter(text):
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=600
    
   
    xt = tokenizer_twitter.texts_to_sequences(text)

    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model_twitter.predict(xt).argmax(axis=1)

    return sentiment_classes[yt[0]]


add_selectbox = st.sidebar.selectbox(
    'Feedback category',
    ['Movies', 'Product','Twitter Analysis']
)
#st.text_input("Your feedback", key="text")

if add_selectbox=='Movies':
	st.title("Welcome to Sentiment analyzer")
	user_review = st.text_input("Your feedback", key="text")
	if(st.button("Analyse")):
		data=[" ".join(preprocess_movies(user_review))]
		st.write(predict_class_movies(data))

elif add_selectbox=='Product':
	st.title("Welcome to Sentiment analyzer")
	user_review = st.text_input("Your feedback", key="text")
	classifier=pipeline("sentiment-analysis")
	if(st.button("Analyse")):
		result=classifier(user_review)
		st.write("Sentiment: "+result[0]['label']+ "\n" + "Confidence:"+ str(result[0]['score']))
	

elif add_selectbox=='Twitter Analysis':
	load_dotenv()
	consumerKey ="2CV30nuFrYXvQszVc618aLS5m"
	consumerSecret = "ceEsspnqKz77i2QBUibCEjfwwIXai199iLFAcwozbTU1BPU7YK"
	accessToken = "3232375908-E7VJTgGjddxMiVWcCTf8ssGTx07Dg9Ey1luNXzU"
	accessTokenSecret = "EAwwlXJ0GreTRHO7WNcI3nzizJHB1l2Rku71fUgqm0MX7"
	#Create the authentication object
	authenticate = tw.OAuthHandler(consumerKey, consumerSecret) 
			
	# Set the access token and access token secret
	authenticate.set_access_token(accessToken, accessTokenSecret) 
			
	# Creating the API object while passing in auth information
	api = tw.API(authenticate, wait_on_rate_limit = True)
	classifier=pipeline("sentiment-analysis")
	def predict_class_twiter(text):
		return (classifier(text))[0]['label']

	def app():
		def process_stauses(sta):
			print(sta.text)

		st.title("Twitter Sentiment Analyzer")

		def load_lottieurl(url: str):
			r = requests.get(url)
			if r.status_code != 200:
				return None
			return r.json()

		lottie_twitter = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_ayl5c9tf.json')
		st_lottie(lottie_twitter, speed=1, height=180, key="initial")

		st.subheader("Analyze Sentiments on Twitter in Real Time!")
		
		st.markdown("Hey there! Welcome to Twitter Sentiment Analysis App. This app scrapes (and never keeps or stores!) the tweets you want to classfiy and analyzes the sentiments as positive, negative or neutral and visualises their distribution.")
		st.markdown("**To begin, please enter the number of tweets you want to analyse.** 👇")

		
		notweet = st.slider('Select a number between 1-100')
		st.write(notweet, 'tweets are being fetched.')
		st.write("__________________________________________________________________________________")

			# Radio Buttons
		st.markdown(" Great! Now, let's select the type of search you want to conduct. You can either search a twitter handle (e.g. @elonmusk) which will analyse the recent tweets of that user or search a trending hashtag (e.g. #WorkFromHome) to classify sentiments of the tweets regarding it. ")
		st.write("")

		stauses = st.radio('Select the mode of fetching',("Fetch the most recent tweets from the given twitter handle","Fetch the most recent tweets from the given twitter hashtag"))

		if stauses == 'Fetch the most recent tweets from the given twitter handle':
			st.success("Enter User Handle")
		elif stauses == 'Fetch the most recent tweets from the given twitter hashtag':
					st.success("Enter Hashtag")
		else:
			st.warning("Choose an option")

			
		raw_text = st.text_input("Enter the twitter handle of the personality (without @) or enter the hashtag (without #)")
		need_help = st.expander('Need help? 👉')
		with need_help:
			st.markdown("Having trouble finding the Twitter profile or Hashtag? Head to the [Twitter website](https://twitter.com/home) and click on the search bar in the top right corner.")

		st.markdown(" ### Almost done! Finally, let's choose what we want to do with the tweets ")
		Analyzer_choice = st.selectbox("Choose the action to be performed 👇",  ["Show Recent Tweets","Classify Sentiment"])


		if st.button("Analyze"):

				
			if Analyzer_choice == "Show Recent Tweets":

				st.success("Fetching latest Tweets")

					
				def Show_Recent_Tweets(raw_text):
						

					if stauses == 'Fetch the most recent tweets from the given twitter handle': 
							posts = [status for status in tw.Cursor(api.user_timeline, screen_name=raw_text,lang='en').items(notweet)]

							
					else :
								posts = [status for status in tw.Cursor(api.search_tweets, q=raw_text,lang='en').items(100)]
									


						
					def get_tweets():

						l=[]
						i=1
						for tweet in posts[:notweet]:
							l.append(tweet.text)
							i= i+1
						return l

					recent_tweets=get_tweets()		
					return recent_tweets

				recent_tweets=Show_Recent_Tweets(raw_text)

				st.write(recent_tweets)
			else:
				st.success("Analysing latest tweets")
				m=[]
				def Analyse_Recent_Tweets(raw_text):
								
					if stauses == 'Fetch the most recent tweets from the given twitter handle': 
							posts = [status for status in tw.Cursor(api.user_timeline, screen_name=raw_text,lang='en').items(notweet)]

							
					else:
						posts=[status for status in tw.Cursor(api.search_tweets, q=raw_text,lang='en').items(100)]

					
					def fetch_tweets():

						l2=[]
						# i=1
						for tweet in posts[:notweet]:
							l2.append(tweet.text)
							# i= i+1
						for j in range(0,notweet):
							#m=[]
							m.append(predict_class_twiter([l2[j]]))
							st.write(l2[j])
							st.write("The predicted sentiment is",predict_class_twiter([l2[j]]))
							st.write("")
							st.write("__________________________________________________________________________________")
							#st.write(m)

					rec_tweets=fetch_tweets()		
					return rec_tweets

				rece_tweets= Analyse_Recent_Tweets(raw_text)

				df = pd.DataFrame(m, columns = ['Sentiment'])
				#st.write(df)
					#df=pandas.DataFrame(m)
				st.markdown("**Whoa! Those are some strong opinions alright. Outta the {0} tweets that we analysed, the positive, negative and neutral sentiment distribution is summed up in the followed visualisation and table.**".format(notweet))
				st.write("")
				fig = px.pie(df,names=df['Sentiment'], title ='Pie chart of different sentiments of tweets')
				st.plotly_chart(fig)
				pos = df[df['Sentiment'] == 'Positive']
					#st.write(pos)
				neg = df[df['Sentiment'] == 'Negative']
					#st.write(neg)
				neu = df[df['Sentiment'] == 'Neutral']
					#st.write(neu)
				total_rows = df.count()
				rowsp = pos.count()
				rowsn = neg.count()
				rowsne = neu.count()
				
					#st.write(total_rows)
				result = pd.concat([rowsp, rowsn, rowsne], axis=1)
				result.columns = ['Positive', 'Negative', 'Neutral' ]
				result.index = ['No. of Tweets']
				st.subheader('Sentiment Distribution')
				st.write(result)
				st.markdown('***')
				st.markdown("Thanks for going through this mini-analysis with us. Cheers!")
					


	if __name__ == "__main__":
		app()
	 
