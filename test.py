import json

tweets_data = []
#tweets_file = open(tweets_data_path, "r")

with open('/Users/uun466/Desktop/Data-Science-Project/DoctorStrange.json') as json_data:
    d = json.loads(json_data, encoding='utf-8')
    if len(d) > 1:
        for each in range(len(d)):
            tweets_data.append(d[each])
    else:
        tweets_data.append(d)
