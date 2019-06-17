#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:43:14 2019

@author: elizabethhutton
"""

import nltk
import pandas as pd
from dataIngest import dataDownload
from dataClean import dataClean, saveClean
from model_lda import LDA, makeDict, saveDict,makeDTM,saveDTM, saveLDA
import argparse

#starting with only AK cases from 2008
api_AK = 'https://api.case.law/v1/cases/?jurisdiction=ark&decision_date_min=2008-12-31&full_case=true'

parser = argparse.ArgumentParser(description='Text 2 Topics with LDA')
parser.add_argument('N_TOPICS', type=int,help='Number of topics to identify (Default=3)')
parser.add_argument('N_PASS', type=int,help='Number of model iterations (Default=10)')

#optional arguments
parser.add_argument('--url', type=str, default=api_AK, help='URL for data download. (Default CAP API)')
                
args = parser.parse_args()
print(args)

#download packages if first time running 
#nltk.download('punkt')
#nltk.download('stopwords')

#define path names
raw_file = 'data/documents_raw.pkl'
clean_file = 'data/documents_clean.pkl' 
model_file = 'data/model.gensim' 
DTM_file = 'data/DTM.pkl'
dict_file = 'data/dictionary.gensim'


#process and save data
url = args.url
dataDownload(url,raw_file)

clean_docs = dataClean(raw_file,FILTER= True,STEM=False)
#saveClean(clean_docs,clean_file)

#make dictionary from documents
dictionary = makeDict(clean_docs)
#saveDict(dict_file)

#make document-term-matrix from docs,dictionary
DTM = makeDTM(clean_docs,dictionary)
#saveDTM(DTM, DTM_file)

##set num topics and LDA iterations
N_TOPICS = args.N_TOPICS
N_PASS = args.N_PASS

#construct LDA model
ldamodel = LDA(DTM,dictionary, N_TOPICS, N_PASS)
saveLDA(ldamodel,model_file)



