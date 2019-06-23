#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:53:00 2019

@author: elizabethhutton
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy


def my_tokenizer(doc):

    """Process document for TF-IDF vectorizer
    
    Keyword Arguments: 
        doc - str, document to process
    Returns: 
        lemmatized_tokens - alphanumeric, lower case, lemmatized tokens

    """
    
    # load spaCy's model for english language
    spacy.load('en')
    
    # instantiate a spaCy tokenizer
    lemmatizer = spacy.lang.en.English()
    
    # apply the preprocessing and tokenzation steps
    tokens = lemmatizer(doc)
    words = [word for word in tokens if word.is_alpha]
    lemmatized_tokens = [token.lemma_ for token in words]
    return lemmatized_tokens
    

def wm2df(wm, feat_names):
    
    """Make pandas dataframe from TF-IDF word matrix"""
    
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return df

def tfidf(raw_text): 
    
    """Apply TF-IDF to documents
    
    Keyword Arguments: 
        raw_text - list of documents and their contents
    Returns: 
        df - pandas dataframe for (doc)x(word) TF-IDF matrix
        tokens - list of token names as strings
        indices - indices sorted from highest to lowest TF-IDF score

    """
    #TF-IDF 
    ###remove words that appear in >80% of docs 
    ###remove english stop words
    ###tokenize according to custom tokenizer function
    vectorizer = TfidfVectorizer(max_df = 0.8,tokenizer=my_tokenizer,stop_words='english')
    wm = vectorizer.fit_transform(raw_text)
    indices = np.argsort(vectorizer.idf_)[::-1]
    tokens = vectorizer.get_feature_names()
    df = wm2df(wm, tokens)
    return df,indices 
    
def filter_by_tfidf(df,indices,top_n, bottom_n):  
    
    """Remove bottom_n words from documents by TF-IDF
    
    Keyword Arguments: 
        df - dataframe output by tfidf 
        indices - indices sorted by decreasing tfidf 
        top_n - int, number of top keywords to return 
        bottom_n - int, number of bottom words to remove
    Returns: 
        filtered - pandas dataframe without low tfidf words
        top_features - list of top_n tokens 
        bottom_features - list of bottom_n tokens (removed from df)
    """
    
    tokens = df.columns
    top_features = [tokens[i] for i in indices[:top_n]]
    bottom_features = [tokens[i] for i in indices[len(tokens)-bottom_n:]]
    filtered = df.drop(bottom_features, axis=1)
    return filtered, top_features, bottom_features



    