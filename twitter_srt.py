# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 00:19:20 2019

@author: Hello
"""

import tweepy #https://github.com/tweepy/tweepy

#Twitter API credentials
consumer_key = "D2CU1ganhMNlIsGg5uSA0vqyg"
consumer_secret = "qtOhQ63NTkHcojQYbj43OPR9FltpBzfXs5fVNlIzYhR3t4tkxF"
access_key = "1164538827777593344-ImvywLNVup4opxe4lGeczxUX6R64Sb"
access_secret = "w1DvHtq2kYVuY2swbJn3f3GtnkPATCm1BsQzLzzoLJdPq"


def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    alltweets = []	
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    
    oldest = alltweets[-1].id-1
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))                # tweet.get('user', {}).get('location', {})
 
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in alltweets]
    
    import pandas as pd
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
    tweets_df.to_csv(screen_name+"_tweets.csv")
    return tweets_df

srt = get_all_tweets("sachin_rt")

import pandas as pd
srt1 =srt.iloc[:,11]
#srt1 =pd.DataFrame(srt1)

srt1 = list(srt1)

srt_rev = ' '.join(srt1)

##Performing tokenization
from nltk.tokenize import sent_tokenize
token_sent = sent_tokenize(srt_rev)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
wordnet = WordNetLemmatizer()
import re

filtered_sent=[]
for i in range(len(token_sent)):
     review = re.sub("[^A-Za-z" "]+"," ",token_sent[i])
     review = re.sub("[0-9" "]+"," ",token_sent[i])
     review = review.lower()
     review =review.split()
     review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
     review = ' '.join(review)
     filtered_sent.append(review)
     
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
text_tf = tf.fit_transform(filtered_sent)
feature_names = tf.get_feature_names()
dense = text_tf.todense()
denselist = dense.tolist()
df =pd.DataFrame(denselist, columns=feature_names)

#plotting wordcloud on TFIDF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
cloud = ' '.join(df)

wordcloud= WordCloud(
        background_color = 'black',
        width = 1800,
        height =1400).generate(cloud)
plt.imshow(wordcloud)

##Importing positive words to plot positive word cloud

with open("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Text mining\\Datasets\\positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")
    
poswords =poswords[36:]

pos_words = ' '.join([w for w in df if w in poswords])

cloud_pos = WordCloud(
        background_color = 'black',
        width =1800,
        height=1400).generate(pos_words)
plt.imshow(cloud_pos)

##Importing negative words to build negative word cloud
with open("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Text mining\\Datasets\\negative-words.txt","r") as nos:
    negwords = nos.read().split("\n")
negwords = negwords[37:]

neg_words = ' '.join([w for w in df if w in negwords])

cloud_neg = WordCloud(
        background_color='black',
        width =1800,
        height =1400).generate(neg_words)
plt.imshow(cloud_neg)
