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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#download packages if first time running 
#nltk.download('punkt')
#nltk.download('stopwords')

#function: dataClean 
#process/tokenize raw text for use with LDA
#inputs: raw_file (str, path of raw text) 
#######  clean_file (str, where to save cleaned text)
#######  STEM (Boolean, true if want to stem words)
def dataClean(raw_file,clean_file,STEM):

    #load data from pickle 
    raw_docs = pd.read_pickle(raw_file)
    raw_text = list(raw_docs['text'])


    stop_words = set(stopwords.words('english'))
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
    
    with open(clean_file, 'wb') as f:
        pickle.dump(clean_docs,f)
    
    return 
    

raw_file = 'documents_raw.pkl'
clean_file = 'documents_clean.pkl' 
dataClean(raw_file,clean_file,STEM=True)

    
    