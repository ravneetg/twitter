
from kafka import KafkaConsumer
import json
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import gensim
import spacy
#import pyspark
#from pyspark import *
import textauger
import re
from textauger import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Vader
from textauger import textfeatures

consumer = KafkaConsumer('Twitter')

#tweets_data = [json.dumps(msg.value) for msg in consumer]

tweets = pd.DataFrame()

tweets_data = []
for msg in consumer:
   tweets_data = msg.value.split(",")
   for each in range(len(tweets_data)):
       #tweets_data
       print tweets_data[each]
       tweets['text'] = map(lambda tweet: tweet['text'].strip(), tweets_data[each])
       tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data[each])
       tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)
       tweets['user_nm'] = map(lambda tweet: tweet['user']['name'].encode('utf-8'), tweets_data)
       tweets['coordinates'] = map(lambda tweet: tweet['coordinates'], tweets_data)
       tweets['location'] = map(lambda tweet: tweet['user']['location'], tweets_data)
       tweets['retweets_count'] = map(lambda tweet: tweet['retweet_count'], tweets_data)
       tweets['text_clean'] = [re.sub(r"http\S+", "", v) for v in tweets.text.values.tolist()]
       tweets['text_clean'] = [re.sub(r"#\S+", "", v) for v in tweets.text_clean.values.tolist()]
       tweets['text_clean'] = [re.sub(r"@\S+", "", v) for v in tweets.text_clean.values.tolist()]
       tweets['text_clean'] = [re.sub(r"u'RT\S+", "", v) for v in tweets.text_clean.values.tolist()]
       tweets['text'] = [v.replace('\n'," ") for v in tweets.text.values.tolist()]
       tweets['text_clean'] = preprocessing.clean_text(text=tweets.text_clean.values,remove_short_tokens_flag=False,lemmatize_flag=True)
       tweets['sentiment_score'] = [textfeatures.score_sentiment(v)['compound'] for v in tweets.text_clean.values.tolist()]
       tweets.loc[tweets['sentiment_score'] > 0.0, 'sentiment'] = 'positive'
       tweets.loc[tweets['sentiment_score'] == 0.0, 'sentiment'] = 'neutral'
       tweets.loc[tweets['sentiment_score'] < 0.0, 'sentiment'] = 'negative'

tweets.to_csv("/Users/uun466/Desktop/Data-Science-Project/tweet_file_new.csv", encoding = 'utf-8')
print tweets
   
