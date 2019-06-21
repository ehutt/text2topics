#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:34 2019

@author: elizabethhutton
"""

import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



def dataClean(raw_file,FILTER,STEM):

    """Return tokenized/processed doc list from raw text.
    
    Tokenize, remove stop words/punctuation/numbers, and stem (optional).
    
    Keyword Arguments: 
        raw_file -- str, path of raw text
        FILTER -- Boolean, true if want to remove stop words
        STEM -- Boolean, true if want to stem words
    Returns: 
        clean_docs list 
    """
    #load data from pickle 
    raw_docs = pd.read_pickle(raw_file)
    raw_text = list(raw_docs['text'])
    
    clean_docs=list() 

    for doc in raw_text: 
        #split into tokens
        tokens = word_tokenize(doc)
        #convert to lower case, remove punctuation (and numbers)
        words = [word.lower() for word in tokens if word.isalpha()]
        if FILTER==True: 
             stop_words = set(stopwords.words('english'))
             words = [w for w in words if not w in stop_words]
        if STEM==True: 
            porter = PorterStemmer()
            words = [porter.stem(w) for w in words]
        clean_docs.append(words)
    
    return clean_docs


def saveClean(clean_docs, clean_file):
    
    """Save clean_docs list to clean_file path"""
    
    with open(clean_file, 'wb') as f:
        pickle.dump(clean_docs,f)
    return

