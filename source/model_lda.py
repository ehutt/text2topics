#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:12:39 2019

@author: elizabethhutton
"""

import pickle
import gensim
from gensim import corpora


#function: LDA
#transforms document to doc-term matrix and apply LDA topic model
#inputs: text_file (str,cleaned text documents)
#######  DTM_file (str, where to save doc-term-matrix)
#######  dict_file (str, where to save dictionary)
#######  model_file (str, where to save lda model)
#######  N_TOPICS (int, num topics to search for)
#######  N_PASS (int, num iterations of LDA)
#returns: ldamodel object 
def LDA(text_file,DTM_file,dict_file,model_file, N_TOPICS, N_PASS):
    
    #for reloading data 
    with open(text_file, 'rb') as file: 
        docs = pickle.load(file)
        
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    
    pickle.dump(doc_term_matrix, open(DTM_file, 'wb'))
    dictionary.save(dict_file)
    
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(doc_term_matrix, num_topics=N_TOPICS, id2word = dictionary, passes=N_PASS)
    ldamodel.save(model_file)
    return ldamodel


#text_file = 'documents_clean.pkl' 
#model_file = 'model1.gensim' 
#DTM_file = 'corpus_matrix.pkl'
#dict_file = 'dictionary.gensim'

#N_TOPICS = 4
#N_PASS = 20

#ldamodel = LDA(text_file,DTM_file,dict_file,model_file, N_TOPICS, N_PASS)

