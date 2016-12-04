from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from kafka import KafkaProducer
#from kafka.client import KafkaClient
from kafka.client import SimpleClient
from kafka.consumer import SimpleConsumer
from kafka.producer import SimpleProducer

import json
import urllib2

# update api url
data = json.load(urllib2.urlopen("http://127.0.0.1:5000/movies"))
movienames= data['moviename'][0]
#print movienames
#### sample data - [u'#moana', u'#doctorstrange', u'#allied', u'#arrivalmovie', u'#badsanta2', u'#almostchristmasmovie']

#***********producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))
client = SimpleClient("localhost:9092")
producer = SimpleProducer(client)

consumer_key = ' ' #'aybpmREJAzUkbrF2f0cWg'#eWkgf0izE2qtN8Ftk5yrVpaaI
consumer_secret = ' '#BYYnkSEDx463mGzIxjSifxfXN6V1ggpfJaGBKlhRpUMuQ02lBX
access_token = ' '#1355650081-Mq5jok7mbcrIbTpqZPcMHgWjcymqSrG1kVaut39
access_token_secret = ' '#QovqxQnw0hSPrKwFIYLWct3Zv4MeGMash66IaOoFyXNWs


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, data):
        producer.send_messages('Twitter', str(data))
        print(data)
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=movienames,languages=["en"])
