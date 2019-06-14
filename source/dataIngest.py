#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:43:41 2019

@author: elizabethhutton
"""

import urllib
import json
import pandas as pd

#PROTOTYPE 

#function: dataDownload 
#downloads JSON data from an API url 
#inputs: api_url (where the data is stored in API) 
#######  saveFile (name of file/path to store extracted raw text)
def dataDownload(api_url, saveFile): 
    
    #download JSON from api url 
    url = urllib.request.urlopen(api_url)
    obj = json.load(url)
    
    data = obj['results']

    #extract nonduplicate opinions from each case 
    docs = set()
    count = 0
    for d in data: 
        if 'casebody' in d: 
            body = d['casebody']
            if 'data' in body: 
                data = body['data']
                if 'opinions' in data: 
                    ops = data['opinions']
                    for op in ops: 
                        if 'text' in op: 
                            docs.add((count,op['text']))
                        else: print("No 'text' key in ops.")
                else: print("No 'opinions' key.")
                count+=1
            else: print("No 'data' key.")
        else: print("No 'casebody' key.")
        
    #convert to dataframe and save
    df = pd.DataFrame(list(docs))
    df.columns = ['case','text']
    df['text'] = df['text'].astype(str)
    df.to_pickle(saveFile)
    return 


#starting with only AK cases from 2008
#api_AK = 'https://api.case.law/v1/cases/?jurisdiction=ark&decision_date_min=2008-12-31&full_case=true'
#raw_file = 'documents_raw.pkl'
#dataDownload(api_AK,raw_file)

            
                
                
        
    
    