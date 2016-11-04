from kafka import KafkaConsumer

consumer = KafkaConsumer('Twitter')

for msg in consumer:
    print (msg.value)
