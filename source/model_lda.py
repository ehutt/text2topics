#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:12:39 2019

@author: elizabethhutton
"""

import pickle
import gensim
from gensim import corpora


def makeDict(clean_docs): 
    
    """Create and return dictionary from clean_docs."""
    
    dictionary = corpora.Dictionary(clean_docs)
    return dictionary


def saveDict(dictionary, dict_file):
    
    """Save dictionary to dict_file path."""
    
    dictionary.save(dict_file)
    return


def makeDTM(clean_docs,dictionary):
    
    """Return doc-term-matrix from clean_docs and dictionary."""
    
    DTM = [dictionary.doc2bow(doc) for doc in clean_docs]
    return DTM

#function: saveDTM
#save DTM to file
#inputs: DTM (dtm obj), DTM_file (str path)
def saveDTM(DTM,DTM_file):
    
    """Save DTM to DTM_file path."""
    
    pickle.dump(DTM, open(DTM_file, 'wb'))
    return 


def LDA(DTM, dictionary, N_TOPICS, N_PASS):
    
    """Create LDA topic model from DTM. 
    
    Keyword Args: 
        DTM -- doc-term-matrix
        dictionary -- gensim dictionary
        N_TOPICS -- int, num topics to search for
        N_PASS -- int, num iterations of LDA
        
    Return: 
        ldamodel object
    """
    
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(DTM, num_topics=N_TOPICS, id2word = dictionary, passes=N_PASS)
    return ldamodel


def saveLDA(ldamodel,model_file):
    
    """Save LDA model to model_file path."""

    ldamodel.save(model_file)
    return
