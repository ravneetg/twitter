
# coding: utf-8

# In[7]:

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import gensim
import spacy
#import pyspark
#from pyspark import *
import textprocessing
import re
from textprocessing import preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer as Vader
from textprocessing import textfeatures
import psycopg2
from os import path
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
from PIL import Image



# In[65]:

d = path.dirname('/Users/uun466/Desktop/Data-Science-Project/')


# In[41]:

#!/usr/bin/python
import sys, os
from PIL import Image
from PIL import ImageFilter
 
inputFile = "test-output-black"
outputFile = os.path.splitext(inputFile)[0] + ".png"
im = Image.open(path.join(d, "thumbs-up-mask.png"))
im = im.filter(ImageFilter.BLUR)
mask = Image.new((im.mode), (im.size))
out = Image.blend(im, mask, 0.3)
out.save(outputFile)


# In[87]:

thumbs_mask_blk = np.array(Image.open(path.join(d, "thumbs-up-hand.jpg")))


# In[88]:

thumbs_mask_blk.ndim


# In[52]:

conn = psycopg2.connect(database="tweetdata", user="postgres", password="pass", host="localhost", port="5432")
dbcur = conn.cursor()
movie_nm = 'Moana'
querystr = dbcur.mogrify("SELECT word, count FROM tweet_words WHERE movie = '%s';" % movie_nm)
print querystr
dbcur.execute(querystr)
freq_dict = []
for record in dbcur:
    freq_dict.append(record)
    
print freq_dict
    


# In[71]:

wc = WordCloud(background_color="white",max_words=1000, mask=thumbs_mask_blk, margin=40,max_font_size=400, 
               relative_scaling= 0.5,random_state=1).generate_from_frequencies(freq_dict)


# In[74]:

wc.words_


# In[93]:

get_ipython().magic(u'matplotlib inline')
wc = WordCloud(background_color="white", max_words=2000, mask=thumbs_mask_blk)
# generate word cloud
wc.fit_words(freq_dict)

# store to file
wc.to_file(path.join(d, "new.png"))

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
#plt.imshow(thumbs_mask_blk, cmap=plt.cm.gray)
#plt.axis("off")
plt.show()


# In[ ]:



