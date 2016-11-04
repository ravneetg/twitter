from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import json
from pandas.io.json import json_normalize
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')
#producer.send('twitterstream', b'some_message_bytes')

#-----------------------------------------------------------------------
# load  API credentials
#-----------------------------------------------------------------------
config = {}
execfile("config.py", config)

#-----------------------------------------------------------------------
# create twitter API object
#-----------------------------------------------------------------------
twitter = Twitter(
		        auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))

#create a producer to write json messages to kafka
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))

#iterator=twitter.search.tweets(q='#KevinHartWhatNow', result_type='recent', lang='en', count=10)
iterator=twitter.search.tweets(q = "#TheGirlOnTheTrain", result_type='recent', lang='en', count = 2)

print "Search complete (%.3f seconds)" % (iterator["search_metadata"]["completed_in"])
#json_normalize(iterator)[["text"]]

for tweet in iterator["statuses"]:
	producer.send('Twitter', tweet)
	print json.dumps(tweet, indent=4)

#print "Result: %s:%s : %s" % (tweet["created_at"],tweet["user"]["screen_name"],tweet["text"])
