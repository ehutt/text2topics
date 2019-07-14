#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:02:21 2019

@author: elizabethhutton
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from yellowbrick.cluster import KElbowVisualizer
from sklearn.neighbors import NearestNeighbors
import spacy
import iterate

class Cluster(): 
    
    def __init__(self,corpus,num_clusters):

        """Perform k-means clustering on corpus.
        
        Keyword Arguments: 
            corpus -- document corpus as Corpus object
            num_clusters -- k clusters to search for
        """
        self.k = num_clusters
        self.top_words = None
        
        word_vectors = corpus.vectors
        kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++')
        self.model = kmeans_clustering
        
        idx = kmeans_clustering.fit_predict(word_vectors)
        self.centers = kmeans_clustering.cluster_centers_
        
        #update corpus vectors with cluster labels
        corpus.clusters = pd.DataFrame(idx,columns=['clusterid'],index=word_vectors.index)
        return 
    
    
    def get_top_words(self, corpus, knn):
        """Get knn top words for each cluster.
        
        Keyword Arguments: 
            corpus -- pandas df of words and their vectors
            knn -- (int) num words to find per cluster
        """
        word_vectors = corpus.vectors
        neigh = NearestNeighbors(n_neighbors=knn, metric= 'cosine')
        neigh.fit(word_vectors)  
        top_word_idxs = list()
        for center in self.centers: 
            center = center.reshape(1,-1)
            top_n = neigh.kneighbors(center,n_neighbors=knn,return_distance=False)
            top_word_idxs.append(top_n)
        
        top_n_words = pd.DataFrame()
        for i, cluster in enumerate(top_word_idxs):
            cluster_name = 'Cluster ' + str(i) 
            words = list() 
            for idx in cluster[0]: 
                word = word_vectors.iloc[idx].name
                words.append(word)
            top_n_words[cluster_name] = words
        self.top_words = top_n_words
        return top_n_words
    
    
    def iterate_kmeans(clean_corpus,elbow):
        #prep for clustering
        clean_corpus.vectorize() 
        
        #iterate kmeans over num topics 
        #methods = 'var','dist','c_h'
        elbow.elbow_kmeans_variance(clean_corpus)
        elbow.elbow_kmeans_inertia(clean_corpus)
        elbow.elbow_kmeans_ch(clean_corpus)
        elbow.elbow_kmeans_dist(clean_corpus)
        return 

    #fix
    def plot_tsne(word_vectors):
        tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
        np.set_printoptions(suppress=True)
        T = tsne.fit_transform(word_vectors)
        labels = word_vectors.index
        
        plt.figure(figsize=(12, 6))
        plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
        for label, x, y in zip(labels, T[:, 0], T[:, 1]):
            plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
        return    
        
def get_kmeans(clean_corpus,num_topics):
    cluster_model = Cluster(clean_corpus,num_topics)
    top_words_kmeans = cluster_model.get_top_words(clean_corpus, knn=10)
    return cluster_model,top_words_kmeans
    
    
    
    
    
    