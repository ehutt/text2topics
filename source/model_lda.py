#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:12:39 2019

@author: elizabethhutton
"""

import pickle
import gensim
from gensim import corpora


#function: makeDict
#creates dictionary from clean_docs
#inputs: clean_docs (cleaned doc list)
#returns: dictionary obj
def makeDict(clean_docs): 
    dictionary = corpora.Dictionary(clean_docs)
    return dictionary

#function: saveDict
#saves dictionary to file
#inputs: dictionary obj
####### dict_file (str, where to save dictionary)
def saveDict(dictionary, dict_file):
    dictionary.save(dict_file)
    return

#function: makeDTM
#makes DTM from clean_doc and dictionary
#inputs: clean_docs (list), dictionary (obj)
#returns: doc-term-mat 
def makeDTM(clean_docs,dictionary):
    DTM = [dictionary.doc2bow(doc) for doc in clean_docs]
    return DTM

#function: saveDTM
#save DTM to file
#inputs: DTM (dtm obj), DTM_file (str path)
def saveDTM(DTM,DTM_file):
    pickle.dump(DTM, open(DTM_file, 'wb'))
    return 

#function: LDA
#transforms document to doc-term matrix and apply LDA topic model
#inputs: text_file (str,cleaned text documents)
#######  DTM (doc-term-matrix)
#######  dictionary (dict obj)
#######  model_file (str, where to save lda model)
#######  N_TOPICS (int, num topics to search for)
#######  N_PASS (int, num iterations of LDA)
#returns: ldamodel object 
def LDA(DTM, dictionary, N_TOPICS, N_PASS):
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(DTM, num_topics=N_TOPICS, id2word = dictionary, passes=N_PASS)
    return ldamodel

#function: saveLDA
#save LDA model
#inputs: ldamodel (LDA obj), model_file (str path)
def saveLDA(ldamodel,model_file):
    ldamodel.save(model_file)
    return

    


