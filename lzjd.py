import pyximport; pyximport.install()
import lzjd_cython
import numpy as np
import os

def hash(b, hash_size=1024):
    if os.path.exists(b): #Was b a path? If it was an valid, lets hash that file!
        #TODO: Add new cython code that reads in a file in chunks and interleaves the digest creation
        in_file = open(b, "rb") # opening for [r]eading as [b]inary
        data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
        in_file.close()
        b = data
    elif isinstance(b, str): #Was a string?, convert to byte array
        b = str.encode(b)
    elif not isinstance(b, bytes):
	    raise ValueError('Input was not a byte array, our could not be converted to one.')

    return lzjd_cython.lzjd_f(b, hash_size)
    
def sim(A, B):
    if isinstance(A, tuple):
        A = A[0]
    if isinstance(B, tuple):
        B = B[0]
    intersection_size = float(np.intersect1d(A, B).shape[0])
    
    #hashes should normally be the same size. Its possible to use different size hashesh tough. 
    #Could happen from small files, or just calling with differen hash_size values
    
    #What if the hashes are different sizes? Math works out that we can take the min length
    #Reduces as back to same size hashes, and its as if we only computed the min-hashing to
    #*just* as many hashes as there were members
    min_len = min(A.shape[0], B.shape[0])
    
    return intersection_size/(2*min_len - intersection_size)
