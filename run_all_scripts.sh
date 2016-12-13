#!/bin/sh
# create directory for installation
mkdir -p  /home/w205user/
cd /home/w205user/

#clone repo
git clone git@github.com:ravneetg/twitter.git

# start api to get movie names
cd /home/w205user/twitter/rest-api
nohup python app.py &

# create database and tables
cd /home/w205user/twitter
createdb -U postgres tweetdata
psql -U postgres -d tweetdata -a -f create_tweets_table.sql
########################################
########## add script to create postgres tables
########################################

# start process to start sourcing twitter data
nohup python extract_tweets_restful.py &

nohup python kafka_Consumer.py &

nohup python extract_tweets_streaming.py &

########################################
###### Add Qian's code#########
#### add script to source data and load tables
######################################## 

########################################
####Add Guillaume's code
########################################

#####################################
