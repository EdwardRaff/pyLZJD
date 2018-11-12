cimport cython
from cpython cimport array
import array
from libc.stdlib cimport malloc, realloc, free
import numpy as np
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

# Helper functions we need

#Rotate bits function
@cython.boundscheck(False)
cdef unsigned int ROTL32(unsigned int x, char r):
    return (x << r) | <int>(<unsigned int>x >> (32 - r)) #similar to >>> in Java

@cython.boundscheck(False)
cdef int fmix32 ( int h ):
    h ^= <int>(<unsigned int>h >> 16) # similar to >>> in Java
    h *= <int>0x85ebca6b
    h ^= <int>(<unsigned int>h >> 13)
    h *= <int>0xc2b2ae35
    h ^= <int>(<unsigned int>h >> 16)
  
    return h


cdef int MurmurHash_PushByte(char b, unsigned int* cur_len, int* _h1, char* data):
    cdef unsigned int _len = cur_len[0]
    
    #store the current byte of input
    data[_len % 4] = b
    cur_len[0] = _len = _len + 1

    cdef int c1 = 0xcc9e2d51
    cdef int c2 = 0x1b873593

    # We will use this as the value of _h1 to dirty for returning to the caller
    cdef int h1_as_if_done
    cdef int k1 = 0
    if _len > 0 and _len % 4 == 0:
        # little endian load order
        k1 = (data[0] & 0xff) | ((data[1] & 0xff) << 8) | ((data[2] & 0xff) << 16) | (data[3] << 24)
        k1 *= c1
        k1 = ROTL32(k1,15)
        k1 *= c2

        _h1[0] = _h1[0] ^ k1
        _h1[0] = ROTL32(_h1[0],13)
        _h1[0] = _h1[0]*5+<int>0xe6546b64
        
        h1_as_if_done = _h1[0]
        # data is out the window now
        data[0] = data[1] = data[2] = data[3] = 0
    else: #Tail case
    
        k1 = (data[0] & 0xff) | ((data[1] & (0xff and _len >= 1) ) << 8) | ((data[2] & (0xff and _len >= 2) ) << 16) | ((data[3] and _len >= 1) << 24)
        h1_as_if_done = _h1[0]

        k1 *= c1
        k1 = ROTL32(k1, 15)
        k1 *= c2
        h1_as_if_done = h1_as_if_done ^ k1
    
    h1_as_if_done = h1_as_if_done ^ <int>_len;

    h1_as_if_done = fmix32(h1_as_if_done);

    return h1_as_if_done;
    
@cython.boundscheck(False)
def lzjd_f(const unsigned char[:] input_bytes, unsigned int hash_size):
    cdef set s1
    s1 = set()
    
    cdef unsigned int cur_length = 0
    cdef char data[4]
    cdef int state = 0
    cdef int hash 
    cdef unsigned int pos = 0
    #Defined b as a char to avoid it becoming a python big-num and causing errors 
    cdef unsigned char b
    cdef unsigned int i
    cdef signed int v

    for b in input_bytes:
        hash = MurmurHash_PushByte(<char>b, &cur_length, &state, data)
        if not hash in s1:
            s1.add(hash)
            #Reset state
            cur_length = 0
            data[0] = data[1] = data[2] = data[3] = 0
            state = 0
    setLength = len(s1)
    
    #Copy set into a new dense array
    cdef signed int* arr = <signed int *>malloc(setLength * cython.sizeof(int))
    for v in s1:
        arr[pos] = v
        pos = pos + 1
    
        
    cdef unsigned int test_size
    if setLength > hash_size:
        #move the smallest k values to front
        q_select(arr, 0, setLength, hash_size)
        test_size = hash_size
        #We don't need anything past the min k, so realloc to free excess memory
        arr =  <signed int *>realloc(arr, hash_size * cython.sizeof(int))
    else:
        test_size = setLength
    sort(arr, test_size)
    
    #O(n) convert to numpy array
    #TODO: Use arrays more efficiently here.
    numpy_arr = np.zeros(shape=(test_size),dtype=np.int32)
    for i in range(test_size):
        numpy_arr[i] = arr[i]
        
    free(arr)
    
    #Can the return type be int*?
    return numpy_arr, setLength
    
#Wrapper function to call qsort
cdef void sort(signed int* y, ssize_t l):
    qsort(y, l, cython.sizeof(int), compare)
    
#Implementation of quick select algorithm
#arr: array of elements
#left: the left most position of the array (inclusive)
#right: the right most position of the array (exclusive)
#k: the rank to get up to 
#NOTE: Normally q_select this naive wouldn't be great. Be we know for fact that we will have random looking values and random distribution b/c all entries are results from hashing. So this should be fine!
cdef q_select(int* arr, int left, int right, int k):
    pivotIndex = q_partition(arr, left, right)
    # The pivot is in its final sorted position
    if k == pivotIndex:
        return arr[k]
    elif k < pivotIndex:
        q_select(arr, left, pivotIndex - 1, k)
    else:
        q_select(arr, pivotIndex + 1, right, k)

#returns position of the pivot
cdef int q_partition(int* arr, int left, int right):
    cdef int pivot = arr[left]
    cdef int temp 
    cdef int i = left - 1
    cdef int j = right + 1
    while True:
        #Find leftmost element greater than or equal to pivot 
        i = i + 1
        while arr[i] < pivot:
            i = i + 1
        #Find rightmost element smaller than  or equal to pivot 
        j = j - 1
        while arr[j] > pivot:
            j = j -1
   
        if i >= j:
            return j
   
        temp = arr[j]
        arr[j] = arr[i]
        arr[i] = temp
        

    
cdef int compare(const_void *va, const_void *vb):
    cdef int a = (<signed int *>va)[0]
    cdef int b = (<signed int *>vb)[0]
    return (a > b) - (a < b)
    