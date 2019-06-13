#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:12:39 2019

@author: elizabethhutton
"""

import pickle
import gensim
from gensim import corpora

filename = 'documents_clean.pkl' 
modelname = 'model1.gensim' 

#for reloading data 
with open(filename, 'rb') as file: 
    docs = pickle.load(file)
    
dictionary = corpora.Dictionary(docs)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]

pickle.dump(doc_term_matrix, open('corpus_matrix.pkl', 'wb'))
dictionary.save('dictionary.gensim')


N_TOPICS = 4
N_PASS = 20
lda = gensim.models.ldamodel.LdaModel
ldamodel = lda(doc_term_matrix, num_topics=N_TOPICS, id2word = dictionary, passes=N_PASS)
ldamodel.save(modelname)




