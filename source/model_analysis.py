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

#TO DO 
#write function to try different topic nums, track coherence 
#try using mallet lda model 
#remove high frequency words 
#t-SNE clustering 





    
