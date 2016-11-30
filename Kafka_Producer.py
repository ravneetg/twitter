
# coding: utf-8

# In[ ]:

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
import textprocessing
import re
from textprocessing import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Vader
from textprocessing import textfeatures


# In[ ]:

producer = KafkaProducer(bootstrap_servers='localhost:9092',api_version=(0,10))


# In[ ]:

consumer_key = "cW3RTKoG5kiNkzfdbSb8aBMyY"
consumer_secret = "iwj5uOncngUMYk2BMNkF3WwL9VR7FXCPvXJVYwbDNDmuy8yRkH"
access_key = "4175914697-j5Ghb209PGZOkobm0cnh9nZ2zMwrrigfVGczYbA"
access_secret = "0lK2XYfh1oksmpymqgTRBLrhR5nGLMUr84N0yhicWuUq2"
#execfile("config_template.py", config)


# In[ ]:

twitter = Twitter(auth = OAuth(access_key, access_secret, consumer_key, consumer_secret))


# In[ ]:

producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))


# In[ ]:

iterator=twitter.search.tweets(q = "#starwars", result_type='recent', lang='en')


# In[ ]:

print "Search complete (%.3f seconds)" % (iterator["search_metadata"]["completed_in"])


# In[ ]:

cnt = len(iterator)

output = []
for tweet in iterator["statuses"]:
        producer.send('Twitter', tweet)
        print tweet
        #print json.dumps(tweet, indent = 4)
        #output.append(tweet)


# In[ ]:

#print output


# In[ ]:

#tweets = pd.DataFrame()


# In[ ]:

#print output


# In[ ]:

#for tweets_data in output[1]:
       #tweets_data
#       print tweets_data
'''
tweets['text'] = map(lambda tweet: tweet['text'], output)
tweets['lang'] = map(lambda tweet: tweet['lang'], output)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, output)
tweets['user_nm'] = map(lambda tweet: tweet['user']['name'].encode('utf-8'), output)
tweets['coordinates'] = map(lambda tweet: tweet['coordinates'], output)
tweets['location'] = map(lambda tweet: tweet['user']['location'], output)
tweets['retweets_count'] = map(lambda tweet: tweet['retweet_count'], output)
tweets['text_clean'] = [re.sub(r"http\S+", "", v) for v in tweets.text.values.tolist()]
tweets['text_clean'] = [re.sub(r"#\S+", "", v) for v in tweets.text_clean.values.tolist()]
tweets['text_clean'] = [re.sub(r"@\S+", "", v) for v in tweets.text_clean.values.tolist()]
tweets['text_clean'] = [re.sub(r"u'RT\S+", "", v) for v in tweets.text_clean.values.tolist()]
tweets['text'] = [v.replace('\n'," ") for v in tweets.text.values.tolist()]
'''


# In[ ]:

'''
tweets['text_clean'] = preprocessing.clean_text(text=tweets.text_clean.values, 
                         remove_short_tokens_flag=False,  
                         lemmatize_flag=True)
'''


# In[ ]:

'''
tweets['sentiment_score'] = [textfeatures.score_sentiment(v)['compound'] for v in tweets.text_clean.values.tolist()]
tweets.loc[tweets['sentiment_score'] > 0.0, 'sentiment'] = 'positive'
tweets.loc[tweets['sentiment_score'] == 0.0, 'sentiment'] = 'neutral'
tweets.loc[tweets['sentiment_score'] < 0.0, 'sentiment'] = 'negative'
'''


# In[ ]:

#tweets.reset_index('text')


# In[ ]:

#producer.send('Twitter', tweets)


# In[ ]:

#tweets


# In[ ]:

#A = tweets.drop()


# In[ ]:

#A = tweets.drop_duplicates(['text'], keep =False)


# In[ ]:

#A.duplicated(['text'])


# In[ ]:

#A


# In[ ]:



