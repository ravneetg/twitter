
drop table if exists tweets;

create table tweets (text varchar(500) PRIMARY KEY NOT NULL,
movie varchar(500),
language varchar(500),
country varchar(500),
user_nm varchar(500),
screen_nm varchar(500),
coordinates_lat varchar(500),
coordinates_long varchar(500),
location varchar(500),
retweets_count INTEGER,
followers_count INTEGER,
favourites_count INTEGER,
friends_count INTEGER,
text_clean varchar(500),
sentiment_score float,
sentiment varchar(500));


drop table if exists tweet_words;

create table tweet_words (word varchar(500) PRIMARY KEY NOT NULL, movie varchar(500),
count integer, word_sentiment varchar(500));
