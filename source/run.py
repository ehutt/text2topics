#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:52:51 2019

@author: elizabethhutton
"""

import corpus
import lda
import clusters
from iterate import Elbow
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import math


#move to config
raw_file = 'data/raw_docs.pkl'
clean_file = 'data/clean_docs.pkl'
stats_file = 'results/corpus_stats.pkl'
results_folder = 'results'
common_stops_file = 'data/common_stops.pkl'
lda_wordcloud_folder = 'results/lda_wordclouds'
kmeans_wordcloud_folder = 'results/kmeans_wordclouds'

def word_cloud(top_words,save_folder):
    
    for i in range(len(top_words)):
        words = list(top_words['Cluster ' + str(i)])
        words = ' '.join(words)
        cloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=3000,
                      height=2000,
                      max_words=10,
                      colormap='tab10',
                      prefer_horizontal=1.0)

        cloud.generate(words)
        plt.gca().imshow(cloud)
        plt.gca().axis('off')
        plt.title('Topic ' + str(i+1))
        cloud.to_file(save_folder + '/topic' + str(i+1) + '.png')
        plt.show()
        #plt.savefig(save_folder + '/topic' + str(i+1))
    return

#clean_corpus = corpus.clean(raw_file,clean_file,stats_file)
#clean_corpus = corpus.load_clean_corpus(clean_file)
#
##define iteration parameters 
#start = 2 
#stop = 50
#step = 2
#elbow = Elbow(start, stop, step, results_folder)
#
##model  
#num_topics = 20 
#lda_model, top_words_lda = lda.get_lda(clean_corpus,num_topics)
word_cloud(top_words_lda,lda_wordcloud_folder)

#cluster_model,top_words_kmeans = clusters.get_kmeans(clean_corpus,num_topics)
#word_cloud(top_words_kmeans,kmeans_wordcloud_folder)



    



