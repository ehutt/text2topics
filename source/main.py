#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:43:14 2019

@author: elizabethhutton
"""

import nltk
from dataIngest import dataDownload
from dataClean import dataClean
from model_lda import LDA

#download packages if first time running 
nltk.download('punkt')
nltk.download('stopwords')

#define variables and path names
#starting with only AK cases from 2008
api_AK = 'https://api.case.law/v1/cases/?jurisdiction=ark&decision_date_min=2008-12-31&full_case=true'
raw_file = 'data/documents_raw.pkl'
clean_file = 'data/documents_clean.pkl' 

#process data
dataDownload(api_AK,raw_file)
dataClean(raw_file,clean_file,STEM=True)

#define vars for LDA
model_file = 'model1.gensim' 
DTM_file = 'corpus_matrix.pkl'
dict_file = 'dictionary.gensim'

N_TOPICS = 4
N_PASS = 20

ldamodel = LDA(clean_file,DTM_file,dict_file,model_file, N_TOPICS, N_PASS)




