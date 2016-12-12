#W205 Class Project
#Analytic Engine for Movies
## Team member: Ravneet Ghuman, Guillaume De Roo, Varadarajan Srinivasan, Qian Yu

## Section 1. Key Files and Descriptions
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
  * get_new_movies.py -- a program to query IMDB restful API with IMDB movie IDs and write data to postgres table, and a .csv file
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
          app.py
    /imdb/
        get_new_movie.py
        search_movie.py
        store_data_hive.sql

## Section 3 1. Pre-requisites
   PYTHON setup
   Download the Python Twitter tools at https://pypi.python.org/pypi/twitter.
   if needed - sudo apt-get install python-setuptools

   python setup.py build
   python setup.py install

   pip install flask
   pip install flask-restful
   pip install sqlalchemy

#### TWITTER setup
   Create an app on https://apps.twitter.com/ and then create auth tokens

#### ZOOKEEPER Setup
   Download zookeeper from
   http://www.apache.org/dyn/closer.cgi/zookeeper/

####start zookeeper
  sudo bin/zkServer.sh start

#### KAFKA Setup
#### Download kafka from
#### https://kafka.apache.org/

####start kafka
   bin/kafka-server-start.sh config/server.properties

#### start kafka-client if you want to view messages are being written
   bin/kafka-console-consumer.sh --zookeeper localhost:2181 --topic Twitter --from-beginning

## Section 3 b. Steps to Run

  Twitter Steaming and Batch Data Ingestion

    1.  Start Kafka zookeeper 
  
      /$path/kafka_2.11-0.10.1.0/bin/zookeeper-server-start.sh /$path/kafka_2.11-0.10.1.0/config/zookeeper.properties
  
    2. Start kafka server
  
      /$path/kafka_2.11-0.10.1.0/bin/kafka-server-start.sh /$path/kafka_2.11-0.10.1.0/config/server.properties
      
    3. Launch API for movie names
       > nohup python app.py &
      Launch kafak producers
    
      Restful Tweets
      
        > nohup python extract_tweets_restful.py &
  
      Streaming Tweets
      
      > nohup python extract_tweets_streaming.py &
      
    4. Define a tweet data table in postgres
    
      > python create_tweets_table.sql
      
    5. Launch kafka consumer
    
      > nohup python kafka_Consumer.py &
  
  IMDB Data Query

    Query IMDB Restful API
    
    1. Install IMDBPy
    
      > pip install imdbpy
   
    2. Query movies with sample program
  
      > python get_new_movies.py 
      
    3. Store data to hive table if needed
    
      > hive -f store_data_hive.sql


    Build IMDB SQL database 
    Note: This is optional only if you want to have a SQL DB of IMDB movies/TV in your own storage, 
          our end to end flow did not use this approach)
          
    1. Download IMDB files at ftp://ftp.fu-berlin.de
    
    2. Install SQLobject object relational manager
    
    3. Create a DB in an SQL database tool such as Postgres
    
    4. Install IMDBPy
    
      >pip install imdbpy
      
    5. Building IMDB relational DB with IMDBPy (This step may take a few hours)
    
      > imdbpy2sql.py -d /IMDB/download_area/ -u 'postgresql://localhost/imdb'
      
    6. Query IMDB SQL database with the example python program
    
      > python get_data_imdbSQL_sample.py
    
    Tableau Dashboard
    
    1. Lauch tableau online
    
    2. Enjoy the dashboard at https://10az.online.tableau.com/#/site/w205movieproject/views
    
## Section 4. Tools and Packages
   
   Kafka version 2.11
   
   Hadoop 2.6
   
   Anaconda Python 2.7
   
   Postgres 9.6
   
   Apache Hive 2.1
   

## Section 5. Known Issues and Challenges

We face several software tool, compatibility and integration issues

1. When we conduct sentiment analysis, we found that python 2.7 NTL package misses Vader Lexicon score file, it needs to be installed manually

2. When using python hiver package to write hive tables, we faced TTransport Exception that prevent the hive table to be written

3. Tableau connection to postgres and hive are challenging mainly due to permission issues

4. It took many hours to built a IMDB SQL database. The schedme from IPDBPy are very convoluted and difficult to use. As a result, we decided to use the restful API.
