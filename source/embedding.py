#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:02:21 2019

@author: elizabethhutton
"""

import spacy 
from spacy import displacy
from clean import dataClean
import pandas as pd


#load pretrained embedding model (GloVe)
#glove = spacy.load('en_vectors_web_lg')
nlp = spacy.load('en_core_web_lg')

raw_file = 'data/documents_raw.pkl'
raw_docs = pd.read_pickle(raw_file)
doc = nlp(raw_docs['text'][0])
 

clean_docs = dataClean(raw_file,FILTER=True,STEM=False)


vectorized_docs = list() 

for i, doc in enumerate(clean_docs):
    doc_vect = pd.DataFrame() 
    for j, word in enumerate(doc):
        v = nlp(word).vector 

