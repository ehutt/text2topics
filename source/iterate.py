#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:43:14 2019

@author: elizabethhutton
"""

from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist,pdist
import numpy as np
import lda 
import clusters


class Elbow():
    
    def __init__(self, start, stop, step, results_folder):
        self.start = start 
        self.stop = stop
        self.step = step
        self.folder = results_folder
        return 
    
    
    def elbow_lda(self, corpus, num_iter):
        
        """Calculate coherence values for a various number of topics.
        
        Keyword Arguments: 
            corpus -- corpus to train on
            num_iter -- int, number of LDA iterations
     
        Returns: 
            model_list (all models tested)
            coherence_values (list of coherence scores)
        """
        coherence_values = []
        model_list = []
        for num_topics in range(self.start, self.stop, self.step):
            print('Topics Tested: ' + str(num_topics)) 
            model = lda.LDA(corpus, num_topics, num_iter)
            model_list.append(model)
            coherence = model.get_coherence_score(corpus)
            coherence_values.append(coherence)
        return model_list, coherence_values
    
    def plot_coherence(self, coherence_values): 
        x = range(self.start, self.stop, self.step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.title('LDA Coherence by # Topics')
        plt.savefig(self.folder + '/coherence_plot.png')
        #plt.show()
        print('Saved coherence plot.')
        return
        
    def elbow_kmeans_variance(self, corpus):
        
        X = corpus.vectors
        KK = range(self.start, self.stop, self.step)
        models = [clusters.Cluster(corpus, num_clusters = k) for k in KK]
        centroids = [km.centers for km in models]
        D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
        dist = [np.min(D,axis=1) for D in D_k]
        
        tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
        totss = sum(pdist(X)**2)/X.shape[0]       # The total sum of squares
        betweenss = totss - tot_withinss          # The between-cluster sum of squares

        # elbow curve
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(KK, betweenss/totss*100, 'b*-')
        ax.plot(KK, betweenss/totss*100, marker='o', markersize=8)
        ax.set_ylim((0,100))
        plt.grid(True)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Percent Variance Explained')
        plt.title('K-Means Elbow Curve')
        plt.savefig(self.folder + '/elbow_variances.png')

        return
        
    def elbow_kmeans_inertia(self, corpus):
        """Perform elbow method for k-means clustering using inertia.
        
        Inertia = Sum of squared distances of samples to their closest 
        cluster center.
        
        Keyword Arguments: 
            corpus -- corpus to train on
        """
        scores = list()
        for k in range(self.start, self.stop, self.step):
            km = KMeans(n_clusters = k, init='k-means++')
            clusters = km.fit(corpus.vectors)
            scores.append(clusters.inertia_)
        x = range(self.start, self.stop, self.step)
        plt.plot(x, scores, marker='o', markersize=8)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Sum of Distances')
        plt.title('K-Means Elbow Curve')
        plt.savefig(self.folder + '/elbow_inertia.png')
        plt.show()
        print('Saved elbow curve.')
        return
        
    def elbow_kmeans_ch(self, corpus):
        """Perform elbow method for k-means clustering using calinski_harabaz.
        
        Keyword Arguments: 
            corpus -- corpus to train on
        """
        km = KMeans(init='k-means++')
        visualizer = KElbowVisualizer(km,k=range(self.start, self.stop, self.step),
                                      metric='calinski_harabaz', timings=False)
        visualizer.fit(corpus.vectors)
        visualizer.poof(outpath= self.folder + '/elbow_c_h.png')
        print('Saved elbow curve.')
        return
    
    def elbow_kmeans_dist(self, corpus):
        """Perform elbow method for k-means clustering using distortion.
        
        Keyword Arguments: 
            corpus -- corpus to train on
        """
        km = KMeans(init='k-means++')
        visualizer = KElbowVisualizer(km,k=range(self.start, self.stop, self.step),
                                      timings=False)
        visualizer.fit(corpus.vectors)
        visualizer.poof(outpath= self.folder + '/elbow_distortion.png')
        print('Saved elbow curve.')
        return