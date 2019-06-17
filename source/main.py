#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:43:14 2019

@author: elizabethhutton
"""

import nltk
from dataIngest import dataDownload
from dataClean import dataClean, saveClean
from model_lda import LDA, makeDict, saveDict,makeDTM,saveDTM, saveLDA

#download packages if first time running 
nltk.download('punkt')
nltk.download('stopwords')


#starting with only AK cases from 2008
api_AK = 'https://api.case.law/v1/cases/?jurisdiction=ark&decision_date_min=2008-12-31&full_case=true'

#define path names
raw_file = 'data/documents_raw.pkl'
clean_file = 'data/documents_clean.pkl' 
model_file = 'model.gensim' 
DTM_file = 'DTM.pkl'
dict_file = 'dictionary.gensim'


#process and save data
dataDownload(api_AK,raw_file)
clean_docs = dataClean(raw_file,STEM=True)
#saveClean(clean_docs,clean_file)

#make dictionary from documents
dictionary = makeDict(clean_docs)
#saveDict(dict_file)

#make document-term-matrix from docs,dictionary
DTM = makeDTM(clean_docs,dictionary)
#saveDTM(DTM, DTM_file)

#set num topics and LDA iterations
N_TOPICS = 4
N_PASS = 20

#construct LDA model
#ldamodel = LDA(DTM,dictionary, N_TOPICS, N_PASS)
#saveLDA(ldamodel,model_file)



