#from twitter_api import *
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import json
from pandas.io.json import json_normalize

#-----------------------------------------------------------------------
# load our API credentials
#-----------------------------------------------------------------------
config = {}
execfile("config.py", config)

#-----------------------------------------------------------------------
# create twitter API object
#-----------------------------------------------------------------------
twitter = Twitter(
		        auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))

#iterator = twitter.search.tweets(q = "lazy dog")

#iterator=twitter.search.tweets(q='#KevinHartWhatNow', result_type='recent', lang='en', count=10)
iterator=twitter.search.tweets(q = "#KevinHartWhatNow", result_type='recent', lang='en', count = 10)

print "Search complete (%.3f seconds)" % (iterator["search_metadata"]["completed_in"])
#json_normalize(iterator)[["text"]]

for tweet in iterator["statuses"]:
    print json.dumps(tweet, indent=4)
    #print "Result: %s:%s : %s" % (tweet["created_at"],tweet["user"]["screen_name"],tweet["text"])
