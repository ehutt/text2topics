#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:34 2019

@author: elizabethhutton
"""

#PROTOTYPE


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
nltk.download('punkt')
nltk.download('stopwords')

#load case data from pickle 
filename1 = 'documents_raw.pkl'
raw_docs = pd.read_pickle(filename1)
raw_text = list(raw_docs['text'])


stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
clean_docs=list() 

for doc in raw_text: 
    #split into tokens
    tokens = word_tokenize(doc)
    #convert to lower case, remove punctuation (and numbers)
    words = [word.lower() for word in tokens if word.isalpha()]
    words_nostops = [w for w in words if not w in stop_words]
    #stemming words potentially removes important semantic info 
    words_stemmed = [porter.stem(w) for w in words_nostops]
    clean_docs.append(words_stemmed)


filename2 = 'documents_clean.pkl' 
with open(filename2, 'wb') as f:
    pickle.dump(clean_docs,f)


    
    