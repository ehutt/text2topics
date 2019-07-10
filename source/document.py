#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:51:45 2019

@author: elizabethhutton
"""

#must download spacy 'en'
import spacy


class Document():
    
    
    def __init__(self, txt):
        if type(txt) != str: 
            print('Expected document as a string!')
            return
        nlp = spacy.load('en')
        tokens = nlp(txt)
        self.raw_count = 0
        self.tokens = tokens
        self.clean_count = 0 
        self.text = txt
        self.topics = None 
        return 
    
    def spacy_tokenize(self):
        nlp = spacy.load('en')
        tokens = nlp(self.text)
        if not type(tokens) == 'spacy.tokens.doc.Doc':
            print('Problem tokenizing text.')
            return
        return tokens
    
    def remove_stops(self,nlp): 
        """Remove common stop words."""
        tokens = nlp(self.text)
        words = [w.text for w in tokens if not w.is_stop]
        self.text = ' '.join(words)
        self.clean_count = len(words)
        return 
    
    def lemmatize(self,nlp):
        """Leammatize tokens."""
        tokens = nlp(self.text)
        words = [w.lemma_ for w in tokens if w.text != '-PRON-']
        self.text = ' '.join(words)
        self.clean_count = len(words)
        return
        
    def remove_names(self,nlp):
        """Remove name entities."""
        tokens = nlp(self.text)
        words = [w.text for w in tokens if w.ent_type_ != 'PERSON']
        self.text = ' '.join(words)
        self.clean_count = len(words)
        return 
    
    def preprocess(self,nlp):
        """Remove special characters and numbers, convert to lowercase."""
        tokens = nlp(self.text)
        words = [w.lower_ for w in tokens if w.is_alpha and not w.is_digit]
        self.text = ' '.join(words)
        self.clean_count = len(words)
        return 
    
   
    
    


        
    
