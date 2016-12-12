#W205 Class Project
#Analytic Engine for Movies
## Team member: Ravneet Ghuman, Guillaume De Roo, Varadarajan Srinivasan, Qian Yu

## Section 1. Files and Descriptions
### Twitter Movie Data
  * extract_tweets_restful.py -- Program that extracts one time batch pull of movie tweets and queues it to Kafka Messaging system
  * extract_tweets_streaming.py -- Program that streams live tweets & sends it to Kafka consumer
  * Kafka_Consumer.py -- Kafka_Consumer program that consumes the tweets both from restful and streaming api. Json response received by the   consumer is analyzed, features extracted, cleansed and loaded into postgres database.
  * create_tweets_table.sql -- SQL script to create the table structures to persist cleansed data from Kafka_Consumer
  preprocessing.py -- Preprocessing program combines several off the shelf functions available in nltk, re, Itertools, Inspect, String etc. into a data cleansing module
  * textfeatures.py -- Textfeatures program combines functions from pandas, gensim and sklearn to create vectorization, feature extraction and sentiment scoring capabilities
  * utils.py -- Utils function offers a logger
  * testapi.py -- Testapi program extracts movie names from hive tables, transforms names into hashtags and passes that onto batch and streaming producer scripts

### IMDB Movie Data
  * get_new_movies.py -- a program to query IMDB restful API with IMDB movie IDs
  * search_movie.py -- a utility program to query IMDB restful API with movie titles

## Section 2. Directory Structure
    /twitter/
      extract_tweets_restful.py
      extract_tweets_streaming.py
      kafka_Consumer.py
      create_tweets_table.sql
      /textprocessing/
          preprocessing.py
          textfeatures.py
          utils.py
    /rest-api/
          testapi.py
    /imdb/
        get_new_movie.py
        search_movie.py
        store_data_hive.sql

## Section 3. Steps to Run

  Twitter Steaming and Batch Data Ingestion

    1.  Start Kafka zookeeper 
  
      /$path/kafka_2.11-0.10.1.0/bin/zookeeper-server-start.sh /$path/kafka_2.11-0.10.1.0/config/zookeeper.properties
  
    2. Start kafka server
  
      /$path/kafka_2.11-0.10.1.0/bin/kafka-server-start.sh /$path/kafka_2.11-0.10.1.0/config/server.properties
      
    3. Launch kafak producers
    
      Restful Tweets
      
        > python extract_tweets_restful.py
  
      Streaming Tweets
      
      > python extract_tweets_streaming.py
      
    4. Define a tweet data table in postgres
    
      > python create_tweets_table.sql
      
    5. Launch kafka consumer
    
      > python kafka_Consumer.py
  
  IMDB Data Query

    Build IMDB SQL database (This is an optional step if you want to have a SQL DB of IMDB movies/TV in your own storage)
    1. Building 
    
    
    
## Section 4. Tools and Packages



## Section 5. Known Issues and Limitations
