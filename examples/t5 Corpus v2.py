#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:11:39 2019

@author: Malware Analysis Lab
"""
# LIBRARIES NEEDED
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyLZJD import digest, sim
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier



# SETTINGS
DATASET_FILE_NAME  = 't5-corpus.zip'
DATASET_FILE_UNZIP = 't5'
DATASET_URL        = 'http://roussev.net/t5/t5-corpus.zip'
TITLE              = 'TSNE Visualization'
PLOT_FILE_NAME     = "t5_perp5.pdf"
PATH_NAME_PATTERN  = "t5/*"
PROCESSES          = -1
X_HASHES           = None


#==============================================================================
# DRIVER
#==============================================================================
def driver():
    
    # download dataset
    download_dataset()
    
    # collect files and their labels
    X_paths, labels_true, Y = collect_files()
    
    # hash, digest files
    hash_files(X_paths)
    
    # get 1D vector
    X = create_1D_vector()
    
    # train the model
    train(X, Y)
    
    # plot the model
    plot(labels_true, X, Y)
 


#==============================================================================
# Downloads the dataset needed
#==============================================================================      
def download_dataset():
    
    # First, lets check if we have the t5 corpus
    if not (os.path.exists(DATASET_FILE_NAME) or
            os.path.exists(DATASET_FILE_UNZIP)):
        print("Downloading ", DATASET_FILE_NAME)
        import urllib.request
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_FILE_NAME)
        except Exception as e:
            print("Error while downloading the dataset")
            print(e)
	
        # Unzip the dataset
        import zipfile
        print("Extracting ", DATASET_FILE_NAME)
        with zipfile.ZipFile(DATASET_FILE_NAME,"r") as zip_ref:
            zip_ref.extractall(".")
        


#==============================================================================
# Collect files from the dataset
#==============================================================================  
def collect_files():
    '''
    Lets collect up all the files in the t5 corpus. Its organized as one big 
    folder, and the extension of each file tells us what kind of file it is. 
    '''
    print("Collecting labels...")
    X_paths = glob.glob(PATH_NAME_PATTERN)
    
    labels_true = list(set([ x[x.find(".")+1:] for x in X_paths]))
    print("Labels:", labels_true)
    
    # Label every file based on which file type it was
    Y = np.asarray([ labels_true.index(x[x.find(".")+1:]) for x in X_paths])
    
    return (X_paths, labels_true, Y)



#==============================================================================
# Take the hash of the files
#============================================================================== 
def hash_files(X_paths):
    global X_HASHES
    '''
    Lets hash all the files now! We have a list of paths, 
    pyLZJD can take that dirrectly and convert it to hashes
    '''
    print("Generating hash of all the files...")
    X_HASHES = digest(X_paths, processes=PROCESSES)
    print("Done hashing!")
    


#==============================================================================
# Creates a vector
#============================================================================== 
def create_1D_vector():
    global X_HASHES
    '''
    We are going to use some tools from scikit-learn. 
    It needs a distance function between data stored as a list of vectors. 
    So we will create a list of 1-D vectors, 
    each each vector sotres the index to it's hash in X_hashes
    '''
    print("Generating 1D vector...")
    X = [[i] for i in range(len(X_HASHES))]
    
    return X



#==============================================================================
# Compute distence between two factors
#============================================================================== 
def lzjd_dist(a, b):
    global X_HASHES
    '''
    Now we define a distance function between two vectors in X. 
    It accesses the index value, and computes the LZJD distance
    '''
    a_i = X_HASHES[int(a[0])]
    b_i = X_HASHES[int(b[0])]
	
    return 1.0-sim(a_i, b_i)



#==============================================================================
# Train the model
#============================================================================== 
def train(X, Y):
    print("Training the ML model...")
    
    knn_model = KNeighborsClassifier(n_neighbors=5, \
                                     algorithm='brute', \
                                     metric=lzjd_dist)
    
    scores = cross_val_score(knn_model, X, Y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), \
                            scores.std() * 2))



#==============================================================================
# Plot the model
#============================================================================== 
def plot(labels_true, X, Y):
    print("Plotting the model...")
    
    X_embedded = TSNE(n_components=2, \
                      perplexity=5, \
                      metric=lzjd_dist).fit_transform(X)
    
    # define the colors to be used in the plot
    colors = [plt.cm.Spectral(each) \
              for each in np.linspace(0, 1, len(labels_true))]
    
    for k, col in zip([z for z in range(len(labels_true))], colors):
    	if k == -1:
    		# Black used for noise.
    		col = [0, 0, 0, 1]
    
    	class_member_mask = (Y == k)
    
    	xy = X_embedded[class_member_mask]
    	plt.plot(xy[:, 0], xy[:, 1], \
              'o', \
              markerfacecolor=tuple(col), \
              markeredgecolor='k', \
              markersize=5, \
              label=labels_true[k])

    plt.title(TITLE)
    plt.legend(loc='upper left')
    plt.savefig(PLOT_FILE_NAME)
    plt.show()
    
    
# START
if __name__ == "__main__":
    driver()
