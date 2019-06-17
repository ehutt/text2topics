#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:51:45 2019

@author: elizabethhutton
"""

#getting a UserWarning for deprecated function smart_open 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

    
import pickle
from matplotlib import pyplot as plt
import gensim
from wordcloud import WordCloud, STOPWORDS
import math


modelname = 'model1.gensim'
corpusname = 'corpus_matrix.pkl' 
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open(corpusname, 'rb'))
ldamodel = gensim.models.ldamodel.LdaModel.load(modelname)
N_TOPICS = 4


#function: coherenceByNumTopics
#computes coherence values for a range of topic numbers
#inputs: DTM (obj)
######## dictionary (obj)
######## clean_docs (list)
######## N_PASS (int, num model iterations)
######## start (int, num topics start), stop (int, max num topics)
######## step (int, num topics step size)
#returns: model_list (all models tested), coherence_values (list of coherence scores)
def coherenceByNumTopics(DTM, dictionary, clean_docs, N_PASS, start=2, stop, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = LDA(DTM,dictionary, num_topics, N_PASS)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=clean_docs, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


    


    
