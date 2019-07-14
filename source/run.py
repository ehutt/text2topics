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
from argparse import ArgumentParser


def word_cloud(top_words,save_folder,show):
    
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
        if show == True:
            plt.show()
    return



def build_parser():
        par = ArgumentParser()
        par.add_argument('--raw_file', type=str,
                         dest='raw_file', help='Filename for loading raw text data.',
                         default="../data/raw_docs.pkl")
        par.add_argument('--clean_path', type=str,
                         dest='clean_path', help='Folder for storing clean data.',
                         default="../data/")
        par.add_argument('--results_path', type=str,
                         dest='results_path', help='Folder to store results.',
                         default="my_model.hdf5")
        par.add_argument('--clean_from_raw', type=int, 
                         dest='clean_from_raw', help='If 1 (default), clean raw text from scratch and store. \
                         If 0, load pre-cleaned data from clean_path.',
                         default=1)
        par.add_argument('--iterate', type=int, 
                         dest='iterate', help='If 1, iterate over different number of topics. \
                         If 0 (default), input desired number of topics.',
                         default=0)
        par.add_argument('--use_lda', type=int, 
                         dest='use_lda', help='If 1, will run LDA instead of clustering + embedding. \
                         If 0 (default), run clustering + embedding model.',
                         default=0)
        return par



if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    raw_file = options.raw_file
    DO_CLEAN = options.clean_from_raw 
    ITERATE = options.iterate
    DO_LDA = options.use_lda
    clean_path = options.clean_path
    results_folder = options.results_path
    
    
    if ITERATE == 1: 
        valid = False 
        while not valid: 
            start = input('Please enter starting # of topics (as integer >= 2): ')
            stop = input('Please enter max # of topics (as integer): ')
            step = input('Please enter the step size (as integer): ')
            try: 
                start = int(start)
                stop = int(stop)
                step = int(step)
                valid = True
            except: 
                valid = False 
        elbow = Elbow(start, stop, step, results_folder)

        if DO_CLEAN == 1: 
            clean_corpus = corpus.clean(raw_file,clean_path,results_folder)
        else:
            clean_corpus = corpus.load_clean_corpus(clean_path)
        
        if DO_LDA == 1: 
            clean_corpus.make_dict()
            clean_corpus.make_dtm()
            model_list, coherence_values = elbow.elbow_lda(clean_corpus,50) #use 50 iterations
            elbow.plot_coherence(coherence_values)
        else:
            clean_corpus.vectorize(clean_path)
            elbow.elbow_kmeans_ch(clean_corpus)
    else: 
        valid = False 
        while not valid: 
            num_topics = input('Please enter desired # of topics (as integer): ')
            try: 
                num_topics = int(num_topics)
                valid = True
            except: 
                valid = False
                
        if DO_CLEAN == 1: 
            clean_corpus = corpus.clean(raw_file,clean_path,results_folder)
        else:
            clean_corpus = corpus.load_clean_corpus(clean_path)
            
        if DO_LDA == 1: 
            clean_corpus.make_dict()
            clean_corpus.make_dtm()
            lda_model, top_words_lda = lda.get_lda(clean_corpus,num_topics)
            word_cloud(top_words_lda,results_folder,show = False)
        else:
            clean_corpus.vectorize(clean_path)
            cluster_model,top_words_kmeans = clusters.get_kmeans(clean_corpus,num_topics)
            word_cloud(top_words_kmeans,results_folder, show = False)


    
    



    



