from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import json
from pandas.io.json import json_normalize
from kafka import KafkaProducer
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


producer = KafkaProducer(bootstrap_servers='localhost:9092',api_version=(0,10))
#producer.send('twitterstream', b'some_message_bytes')

#-----------------------------------------------------------------------
# load  API credentials
#-----------------------------------------------------------------------
consumer_key = "cW3RTKoG5kiNkzfdbSb8aBMyY"
consumer_secret = "iwj5uOncngUMYk2BMNkF3WwL9VR7FXCPvXJVYwbDNDmuy8yRkH"
access_key = "4175914697-j5Ghb209PGZOkobm0cnh9nZ2zMwrrigfVGczYbA"
access_secret = "0lK2XYfh1oksmpymqgTRBLrhR5nGLMUr84N0yhicWuUq2"
#execfile("config_template.py", config)

#print config
#-----------------------------------------------------------------------
# create twitter API object
#-----------------------------------------------------------------------
twitter = Twitter(auth = OAuth(access_key, access_secret, consumer_key, consumer_secret))

#create a producer to write json messages to kafka
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))

#iterator=twitter.search.tweets(q='#KevinHartWhatNow', result_type='recent', lang='en', count=10)
iterator=twitter.search.tweets(q = "#TheGirlOnTheTrain", result_type='recent', lang='en')

print "Search complete (%.3f seconds)" % (iterator["search_metadata"]["completed_in"])
#json_normalize(iterator)[["text"]]

cnt = len(iterator)

output = []
for tweet in iterator["statuses"]:
	producer.send('Twitter', tweet)
	#print json.dumps(tweet, indent = 4)
        output.append(tweet)

#with open('tweet_file.json', 'w') as f:
#     json.dump(output, f)

print output
print len(output)

tweets = pd.DataFrame()

for tweets_data in output:
       #tweets_data
       print tweets_data
       tweets['text'] = map(lambda tweet: tweet['text'].strip(), tweets_data)
       tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
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

#tweets.to_csv("/Users/uun466/Desktop/Data-Science-Project/tweet_file_new.csv", encoding = 'utf-8')

print tweets

