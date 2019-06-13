#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:51:45 2019

@author: elizabethhutton
"""

#getting a UserWarning for deprecated function smart_open 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

    
import pickle
from matplotlib import pyplot as plt
import gensim
from gensim import corpora
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import math
import pyLDAvis.gensim


modelname = 'model1.gensim'
corpusname = 'corpus_matrix.pkl' 
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open(corpusname, 'rb'))
ldamodel = gensim.models.ldamodel.LdaModel.load(modelname)

#function: topicCloud
#generates word clouds for each topic
#inputs: model (obj, LDA model)
#######  N_TOPICS (int, num topics)
#returns: matplotlib pyplot
def topicCloud(model,N_TOPICS): 
    
    cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      prefer_horizontal=1.0)
    
    topics = model.show_topics(formatted=False)
    h = math.ceil(N_TOPICS/2)
    fig, axes = plt.subplots(h, 2, figsize=(10,10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    return plt 





    
