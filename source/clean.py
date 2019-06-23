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
from nltk.stem import WordNetLemmatizer


def sort_common_words(text, ngram = 1, n = 10,  tfidf = False, stopwords = None):
    '''
    Return incidence of the n words that appear in the highest proportion of text samples
    
    Input:
        text (pd.Series): text to be analyzed
        n-gram(int): n-gram to analyze (default: unigram)
        n (int): number of words to return
        tfidf (boolean): if True, use tf-idf vector instead of binary count
        stopwords (str or None): common/basic words to remove in the vectorization process (keep None if using to find stopwords) 
    Output:
        top_n_words: proportion of text samples that contain each of the top n words
    Credit: Edwin from Ping
    '''
    from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
    import pandas as pd
    
    # Transform data into vectorized word binary counts or tf-idf counts
    if tfidf == True: 
        vect =  TfidfVectorizer(lowercase=True, analyzer='word', ngram_range=(ngram, ngram), stop_words = stopwords)
    else:
        vect = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(ngram, ngram), stop_words = stopwords)
    word_counts = vect.fit_transform(text)
    vocab = vect.get_feature_names()
    num_entries = word_counts.shape[0]
    
    # Convert sparse matrix to a 1-column pandas DataFrame then to a pandas Series
    word_counts = word_counts.sum(axis = 0)
    word_counts = pd.DataFrame(word_counts)
    word_counts.columns = vocab
    word_counts = word_counts.transpose()
    word_counts = word_counts.iloc[:, 0]
    
    # Sort by word's prevalence and convert to proportion of text entires that includes the word
    top_n_words = word_counts.nlargest(n) / num_entries
    
    return top_n_words

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
    wordnet_lemmatizer = WordNetLemmatizer()

    for doc in raw_text: 
        #split into tokens
        tokens = word_tokenize(doc)
        #convert to lower case, remove punctuation (and numbers)
        words = [word.lower() for word in tokens if word.isalpha()]
        words = [wordnet_lemmatizer.lemmatize(word) for word in words]
        if FILTER==True: 
             stop_words = stopwords.words('english')
             extra_stops = ['also','would','case','id','therefore','upon','within']
             stop_words.extend(extra_stops)
             words = [w for w in words if not w in stop_words]
             #remove high frequency words 
             legal_stops = sort_common_words(words,n=30)
             words = [w for w in words if not w in legal_stops]
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





    
    