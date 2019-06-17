#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:34 2019

@author: elizabethhutton
"""


#Notes: 
#consider identifying POS (at least for names) to avoid case problems
### e.g. PunktSentenceTokenizer
#to stem or not to stem? 

import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer




def dataClean(raw_file,STEM):

    """Return tokenized/processed doc list from raw text.
    
    Tokenize, remove stop words/punctuation/numbers, and stem (optional).
    
    Keyword Arguments: 
        raw_file -- str, path of raw text
        STEM -- Boolean, true if want to stem words
    Returns: 
        clean_docs list 
    """
    
    #load data from pickle 
    raw_docs = pd.read_pickle(raw_file)
    raw_text = list(raw_docs['text'])

    stop_words = set(stopwords.words('english'))
    #stop_words.append('court')
    porter = PorterStemmer()
    clean_docs=list() 

    for doc in raw_text: 
        #split into tokens
        tokens = word_tokenize(doc)
        #convert to lower case, remove punctuation (and numbers)
        words = [word.lower() for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        if STEM==True: 
            words = [porter.stem(w) for w in words]
        clean_docs.append(words)
    
    return clean_docs


def saveClean(clean_docs, clean_file):
    
    """Save clean_docs list to clean_file path"""
    
    with open(clean_file, 'wb') as f:
        pickle.dump(clean_docs,f)
    return 





    
    