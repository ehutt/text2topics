#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:43:14 2019

@author: elizabethhutton
"""

import ingest
import clean 
import model_lda 
import pandas as pd

#import argparse

#starting with only AK cases from 2008
api_AK = 'https://api.case.law/v1/cases/?jurisdiction=ark&decision_date_min=2008-12-31&full_case=true'


#parser = argparse.ArgumentParser(description='Text 2 Topics with LDA')
#parser.add_argument('N_TOPICS', type=int,help='Number of topics to identify (Default=3)')
#parser.add_argument('N_PASS', type=int,help='Number of model iterations (Default=10)')

#optional arguments
#parser.add_argument('--url', type=str, default=api_AK, help='URL for data download. (Default CAP API)')
#                
#args = parser.parse_args()
#print(args)

##set num topics and LDA iterations
#N_TOPICS = args.N_TOPICS
#N_PASS = args.N_PASS


#define path names
raw_file = 'data/documents_raw.pkl'
clean_file = 'data/documents_clean.pkl' 
model_file = 'data/model.gensim' 
DTM_file = 'data/DTM.pkl'
dict_file = 'data/dictionary.gensim'


#process and save data
url = api_AK
#url = args.url
ingest.dataDownload(url,raw_file)
raw_docs = pd.read_pickle(raw_file)
raw_text = list(raw_docs['text'])
clean_docs = clean.dataClean(raw_file,FILTER= True,STEM=False)

#make dictionary from documents
dictionary = model_lda.makeDict(clean_docs)

#make document-term-matrix from docs,dictionary
DTM = model_lda.makeDTM(clean_docs,dictionary)

N_PASS = 30
N_TOPICS = 12

#construct LDA model
#ldamodel = model_lda.LDA(DTM,dictionary, N_TOPICS, N_PASS)
#model_lda.saveLDA(ldamodel,model_file)

#model_vis.topicCloud(ldamodel,N_TOPICS)



