#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:50:34 2019

@author: elizabethhutton
"""



 #load pretrained embedding model ('en_core_web_lg')

from gensim import corpora
import nltk
from nltk.tokenize import word_tokenize
import numpy as np 
import pandas as pd
import spacy
import pickle
from document import Document
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer


class Corpus(): 
    
    def __init__(self, documents):
        if type(documents) != list: 
            print('Expected documents as a list!')
            return
        self.dictionary = None
        self.dtm = None
        self.tfidf = None
        self.word_freq = None
        self.docs = documents
        self.tokens = self.nltk_tokenize()
        self.vectors = None
        return 
    
    def nltk_tokenize(self):
        tokenized_docs = list()
        for doc in self.docs: 
            tokens = word_tokenize(doc.text)
            tokenized_docs.append(tokens)
        return tokenized_docs
            
    def make_dict(self):
        if type(self.tokens[0]) != list: 
            print('Expected documents as list of tokens!')
        self.dictionary = corpora.Dictionary(self.tokens)
        return 
    
    def make_dtm(self):
        if type(self.tokens[0]) != list: 
            print('Expected documents as tokens!')
        self.dtm = [self.dictionary.doc2bow(doc) for doc in self.tokens]
        return  
    
    def save_corpus_text(self, file_name):
        doc_list = list()
        with open(file_name, 'wb') as f:
            for doc in self.docs: 
                doc_list.append(doc.text)
            pickle.dump(doc_list,f)
        f.close()
        print('Saved clean documents to '+ file_name)
        return 
    
    def save_corpus_stats(self,stats_file):
        if self.avg_raw_word_count() == 0 or None: 
            print('Word count None or 0.')
            return
        raw_avg = self.avg_raw_word_count()
        clean_avg = self.avg_clean_word_count()
        with open(stats_file, 'wb') as f:
            pickle.dump(raw_avg,f)
            pickle.dump(clean_avg,f)
        f.close()
        return
        
        
    def avg_raw_word_count(self):
        counts = list()
        for doc in self.docs: 
            counts.append(doc.raw_count)
        avg = sum(counts)/len(counts)
        return avg 
    
    def avg_clean_word_count(self):
        counts = list()
        for doc in self.docs: 
            counts.append(doc.clean_count)
        avg = sum(counts)/len(counts)
        return avg 
    
    def common_words(self, n_words):
        text = list() 
        for doc in self.docs: 
            text.append(doc.text)
        texts = pd.Series(text)
        
        vect = CountVectorizer(lowercase=True, stop_words = 'english')
        word_counts = vect.fit_transform(texts)
        vocab = vect.get_feature_names()
        num_entries = word_counts.shape[0]
        
        # Convert sparse matrix to a 1-column pandas DataFrame then to a pandas Series
        word_counts = word_counts.sum(axis = 0)
        word_counts = pd.DataFrame(word_counts)
        word_counts.columns = vocab
        word_counts = word_counts.transpose()
        word_counts = word_counts.iloc[:, 0]
        
        # Sort by word's prevalence and convert to proportion of text entires that includes the word
        top_n_words = word_counts.nlargest(n_words) / num_entries
        
        return top_n_words
    
    
    def remove_common_words(self, n_words):
        """Remove n extra words based on raw count."""

        extra_stops = self.common_words(n_words)
        extra_stops = list(extra_stops.index)
        extra_stops.append('-PRON-')
        clean_docs=list()
            
        for doc in self.tokens: 
            words = [w for w in doc if w not in extra_stops]
            text = ' '.join(words)
            document = Document(text)
            clean_docs.append(document)
            
        new_corpus = Corpus(clean_docs)
        return new_corpus, extra_stops
        
        

    
    def get_tfidf():
        return 
    
    def get_frequency(): 
        return
    
    def vectorize(self): 
        
        """Get GloVe representations for unique words in corpus"""
        
        #load pretrained embedding model (GloVe)
        glove = spacy.load('en_core_web_lg')
        #extract unique words (aka vocabulary)
        unique_words = set()
        for d in self.docs: 
            txt = d.text
            doc = glove(txt)
            for word in doc: 
                if word.has_vector:
                    unique_words.add(word.text)
        #change set to list type
        unique_words = list(unique_words)
        #save vector representation
        word_vectors = np.array([glove(word).vector for word in unique_words if glove(word).has_vector])
        #index vectors by corresponding word 
        corpus_vectors = pd.DataFrame(word_vectors, index=unique_words)
        self.vectors = corpus_vectors
        return
    
def load_raw_text(raw_file):
    raw = pd.read_pickle(raw_file)
    raw_text = list(raw['text'])
    return raw_text

def load_clean_corpus(clean_path):
    clean_file = clean_path + 'clean.pkl'
    clean_text = pickle.load( open(clean_file, "rb" ) )
    clean_docs = list()
    for text in clean_text: 
        doc = Document(text)
        clean_docs.append(doc)
    clean_corpus = Corpus(clean_docs)
    print('Loaded clean docs.')
    return clean_corpus

def clean(raw_file,clean_path,results_path):
    """Preprocess documents
    
    Convert to lower-case.
    Remove punctuation, numbers, special chars.
    Remove name entities. 
    Remove common stop words.
    Lemmatize.
    Get stats like avg dirty vs. clean word count.
    
    Save text to clean_file and stats to stats_file.
    Return clean_corpus as Corpus object.
    """
    clean_file = clean_path + 'clean.pkl'
    stats_file = results_path + 'stats.pkl' 
    raw_text = load_raw_text(raw_file)    
    clean_docs = list()
    nlp = spacy.load('en')
    i = 0
    print('Cleaning documents...')
    for text in raw_text: 
        words = nlp(text)
        raw_count = len(words)
        words = [w for w in words if not w.is_stop]
        words = [w for w in words if w.ent_type_ != 'PERSON' and w.pos_ != 'PROPN']
        words = [w for w in words if w.is_alpha and not w.is_digit]
        words = [w.lemma_ for w in words if w.text != '-PRON-']
        word_string = ' '.join(words)
        word_string = word_string.lower()
        doc = Document(word_string)
        doc.clean_count = len(words)
        doc.raw_count = raw_count
        clean_docs.append(doc)
        if i%10 == 0:
            print('Document: ' + str(i))
        i += 1
    clean_corpus = Corpus(clean_docs)
    clean_corpus.save_corpus_text(clean_file)
    clean_corpus.save_corpus_stats(stats_file)
    print('Done')
    return clean_corpus


    