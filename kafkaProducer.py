
from kafka.client import KafkaClient
from kafka.producer import SimpleProducer
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')
producer.send('twitterstream', 'Hello123')

kafka =  KafkaClient("localhost:9092")

producer = SimpleProducer(kafka)

producer.send_messages("pythontest", "This is message sent from python client " + str(datetime.now().time()) )

producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))
producer.send('twitterstream', {'foo': 'bar'})
###############33

# encode objects via msgpack
producer = KafkaProducer(value_serializer=msgpack.dumps)
producer.send('twitterstream', {'foo': 'bar'})



consumer = KafkaConsumer('my_favorite_topic')
for msg in consumer:
    print (msg)
