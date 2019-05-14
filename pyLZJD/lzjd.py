import numpy as np

import pyximport; 
pyximport.install(setup_args={"include_dirs":np.get_include()})
from . import lzjd_cython

import os
from multiprocessing import Pool 
import functools
import scipy


def isFile(s):
    try:
        return isinstance(s, str) and os.path.isfile(s)
    except:
        return False

def digest(b, hash_size=1024, mode=None, processes=-1, false_seen_prob=0.0):
    if isinstance(b, list): #Assume this is a list of things to hash. 
        mapfunc = functools.partial(digest, hash_size=hash_size, mode=mode, false_seen_prob=false_seen_prob)
        if processes < 0:
            processes = None
        elif processes <= 1: # Assume 0 or 1 means just go single threaded
            return [z for z in map(mapfunc, b)]
        #Else, go multi threaded!
        pool = Pool(processes)
        to_ret = [z for z in pool.map(mapfunc, b)]
        pool.close()
        return to_ret
    #Not a list, assume we are processing a single file
    if isFile(b): #Was b a path? If it was an valid, lets hash that file!
        #TODO: Add new cython code that reads in a file in chunks and interleaves the digest creation
        in_file = open(b, "rb") # opening for [r]eading as [b]inary
        data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
        in_file.close()
        b = data
    elif isinstance(b, str): #Was a string?, convert to byte array
        b = str.encode(b)
    elif not isinstance(b, bytes):
        raise ValueError('Input was not a byte array, our could not be converted to one.')

    if mode == "SuperHash" or mode == "sh":
        return lzjd_cython.lzjd_fSH(b, hash_size, false_seen_prob)
    return lzjd_cython.lzjd_f(b, hash_size, false_seen_prob)
    
def sim(A, B):
    if isinstance(A, tuple):
        A = A[0]
    if isinstance(B, tuple):
        B = B[0]
    
    #What type of hash did we use? If its a np.float32, we did SuperHash
    if A.dtype == np.float32:
        return np.sum(A == B)/A.shape[0]
    #Else, we are doing the normal case of set intersection
    
    intersection_size = lzjd_cython.intersection_size(A, B)
    #intersection_size = float(np.intersect1d(A, B, assume_unique=True).shape[0])
    
    #hashes should normally be the same size. Its possible to use different size hashesh tough. 
    #Could happen from small files, or just calling with differen hash_size values
    
    #What if the hashes are different sizes? Math works out that we can take the min length
    #Reduces as back to same size hashes, and its as if we only computed the min-hashing to
    #*just* as many hashes as there were members
    min_len = min(A.shape[0], B.shape[0])
    
    return intersection_size/float(2*min_len - intersection_size)

def vectorize(b, hash_size=1024, k=8, processes=-1, false_seen_prob=0.0):
    if isinstance(b, list): #Assume this is a list of things to hash. 
        mapfunc = functools.partial(vectorize, hash_size=hash_size, k=k, false_seen_prob=false_seen_prob)
        if processes < 0:
            processes = None
        elif processes <= 1: # Assume 0 or 1 means just go single threaded
            return [z for z in map(mapfunc, b)]
        #Else, go multi threaded!
        pool = Pool(processes)
        to_ret = [z for z in pool.map(mapfunc, b)]
        pool.close()
        return scipy.sparse.vstack(to_ret) #Make it into one big matrix please, k-thnx
    
    #Not a list, assume we are processing a single file
    if isFile(b): #Was b a path? If it was an valid, lets hash that file!
        #TODO: Add new cython code that reads in a file in chunks and interleaves the digest creation
        in_file = open(b, "rb") # opening for [r]eading as [b]inary
        data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
        in_file.close()
        b = data
    elif isinstance(b, str): #Was a string?, convert to byte array
        b = str.encode(b)
    elif isinstance(b, np.ndarray): #Is a numpy array? Thats OK if its a float - we assume its already been hashed
        if b.dtype != np.float32:
            raise ValueError('Input was not a byte array, or a SuperMinHash, our could not be converted to one.')
    elif isinstance(b, tuple) and len(b) == 2: #This might be a raw output from hash, lets pass the first value through again
        return vectorize(b[0])
    elif not isinstance(b, bytes):
        raise ValueError('Input was not a byte array, our could not be converted to one.')
    
    #OK, its now either bytes of np.float32. If bytes, make it a np.float32
    if isinstance(b, bytes):
        b = digest(b, hash_size=hash_size, mode="sh", false_seen_prob=false_seen_prob)[0]
        
    #OK, now its defintly a np.float32, lets convert to feature vector!
    return lzjd_cython.k_bit_float2vec(b, k)
