#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:43:41 2019

@author: elizabethhutton
"""

import urllib
import json
import pandas as pd
import lzma 


#function: dataDownload 

def dataDownload(api_url, saveFile): 
    
    """Download JSON data from an CAP API url. 
    
    Extract opinion text only and saves to file. 
    
    Keyword Args: 
        api_url -- str, where the data is stored in API
        saveFile str, path to store extracted raw text
    """
    
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


raw_file = '../data/raw/Arkansas/data/' 
            
                
                
        
    
    