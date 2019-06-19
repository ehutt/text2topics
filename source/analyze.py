#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:51:45 2019

@author: elizabethhutton
"""

#getting a UserWarning for deprecated function smart_open 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

    
from matplotlib import pyplot as plt
import model_lda
from gensim.models import CoherenceModel


def coherenceByNumTopics(DTM, dictionary, clean_docs, N_PASS, stop, start=2, step=3):
    
    """Calculate coherence values for a various number of topics.
    
    Keyword Arguments: 
        DTM -- doc-term matrix
        dictionary -- gensim dictionary
        clean_docs -- list of tokenized documents
        N_PASS -- int, number of LDA iterations
        start -- int, initial num topics
        end -- int, max num topics 
        step -- step size 
    Returns: 
        model_list (all models tested)
        coherence_values (list of coherence scores)
    """
    
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = model_lda.LDA(DTM,dictionary, num_topics, N_PASS)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=clean_docs, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def plotCoherence(coherence_values, start, stop, step): 
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return


    


    
