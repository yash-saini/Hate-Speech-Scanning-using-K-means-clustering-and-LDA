# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:06:22 2020

@author: YASH
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:41:36 2020

@author: YASH
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:22:06 2019

@author: YASH
"""
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
import csv
import pandas as pd
import numpy 
import nltk

#Data cleaning Functions:
def isEnglish(s):
    try:
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True

    #The following function removes the part of the string that contains the substring eg. if
    #substring = 'http' , then http://www.google.com is removed, that means, remove until a space is found
def rem_substring(tweets,substring):
    m=0;
    for i in tweets:
        if (substring in i):
        #while i.find(substring)!=-1:
            k=i.find(substring)
            d=i.find(' ',k,len(i))
            if d!=-1:               #substring is present somwhere in the middle(not the end of the string)
                i=i[:k]+i[d:]
            else:                   #special case when the substring is present at the end, we needn't append the
                i=i[:k]             #substring after the junk string to our result
        tweets[m]=i #store the result in tweets "list"
        m+= 1
    return tweets
def removeNonEnglish(tweets):
    result=[]
    for i in tweets:
        if isEnglish(i):
            result.append(i)
    return result

#the following function converts all the text to the lower case
def lower_case(tweets):
    result=[]
    for i in tweets:
        result.append(i.lower())
    return result

def rem_punctuation(tweets):
    #print(len(tweets))
    m=0
    validLetters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    for i in tweets:
        x = ""
        for j in i:
            if (j in validLetters)==True:
                x += j
        tweets[m]=x
        m=m+1
    return tweets

def stop_words(tweets):
    #Removal of Stop words like is, am , be, are, was etc.
    stop_words1 = set(stopwords.words('english')) 
    indi=0
    for tweet in tweets:
        new_s=[]
        Br_tweet = word_tokenize(tweet)
        for word in Br_tweet:
            if (word not in stop_words1):
                new_s.append(word)
        et=" ".join(new_s)
        tweets[indi]=et
        indi+=1
    return tweets
        
                

 #POS Tagger Function used to identify the adjectives, verbs, adverbs.

def POS_tagger(tweets, username):
    final = []
        # for each line in tweets list
    for line in tweets:
        t = []
            # for each sentence in the line
            # tokenize this sentence
        text= word_tokenize(line)
       
        k = nltk.pos_tag(text)
        for i in k:
                # Only Verbs, Nouns Adverbs & Adjectives are Considered
                if ((i[1][:2] == "VB") or (i[1][:2] == "JJ") or (i[1][:2] == "RB") or (i[1][:]=="NN") or (i[1][:]=="NNS")):
                    t.append(i[0])
        one_tweet=" ".join(t)
        if (len(one_tweet)>0):
            final.append(one_tweet)
    final=lower_case(final)
    dict1={'POS_Tweet':final}
    db1=pd.DataFrame(dict1)
    filename = "Pos_tagged_" + username + ".csv"
    
    db1.to_csv(filename)

def stemming(word):
    # Find the root word
    # stemming of words
    porter = PorterStemmer()
    stemmed = porter.stem(word) 
    return stemmed

def main():
    c=raw_input("Enter the name of the tweet file:")
    c_f=c+'.csv'
    db=pd.read_csv(c_f)
    tweets=list(db['tweet'])
    tweets=rem_substring(tweets,'#')
    tweets=rem_substring(tweets,'http')
    tweets=rem_substring(tweets,'https')
    tweets=rem_substring(tweets,'www')
    tweets=rem_substring(tweets,'@')
    tweets=rem_substring(tweets,'RT')
    
    tweets=rem_punctuation(tweets)
    tweets=stop_words(tweets)
    tweets= removeNonEnglish(tweets)
   
   
    

    
    #tweets.replace("."," ")
    for tweet in tweets:
        tweet=tweet.replace("."," ")

    ''' dict1={'Tweet':tweets}
    db1=pd.DataFrame(dict1)
    r_f='cleaned_'+ c + '.csv'
    db1.to_csv(r_f)
    '''    

    
    POS_tagger(tweets,c)
     
    print("Tweets have now been cleaned !!")

main()