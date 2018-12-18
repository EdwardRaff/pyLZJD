cimport cython
from cpython cimport array
import array
from libc.stdlib cimport malloc, realloc, free
import numpy as np
cimport numpy as np
from math import floor
from scipy.sparse import csr_matrix 




#We are going to get a warning from cython that looks like 
#warning: "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
#We can ignore it https://stackoverflow.com/questions/25789055/cython-numpy-warning-about-npy-no-deprecated-api-when-using-memoryview

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
cdef computeLZset(const unsigned char[:] input_bytes, unsigned int hash_size):
    """
    This method does the lifting to compute the LZ set using the LZJD_f variant of LZJD. 
    """
    cdef set s1
    s1 = set()
    
    cdef unsigned int cur_length = 0
    cdef char data[4]
    cdef int state = 0
    cdef int hash 
    #Defined b as a char to avoid it becoming a python big-num and causing errors 
    cdef unsigned char b

    for b in input_bytes:
        hash = MurmurHash_PushByte(<char>b, &cur_length, &state, data)
        if not hash in s1:
            s1.add(hash)
            #Reset state
            cur_length = 0
            data[0] = data[1] = data[2] = data[3] = 0
            state = 0
    return s1
    
@cython.boundscheck(False)
def lzjd_f(const unsigned char[:] input_bytes, unsigned int hash_size):
    """
    This method computes the LZJD set using the original paper's approach. 
    We find the LZJD_f set, and find the k small hash values (k=hash_size). 
    We then return that list as the LZJD digest
    """
    cdef set s1 = computeLZset(input_bytes, hash_size)
    
    cdef unsigned int pos = 0
    cdef unsigned int i
    cdef signed int v
    cdef unsigned int setLength = len(s1)
    
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
        #arr =  <signed int *>realloc(arr, hash_size * cython.sizeof(int))
    else:
        test_size = setLength
    sort(arr, test_size)
    
    #O(n) convert to numpy array
    #TODO: Use arrays more efficiently here.
    numpy_arr = np.zeros(shape=(test_size),dtype=np.int32)
    for i in range(test_size):
        numpy_arr[i] = arr[i]
        
    free(arr)
    arr = NULL
    
    #Can the return type be int*?
    return numpy_arr, setLength


@cython.boundscheck(False)
@cython.wraparound(False)   # Deactivate negative indexing.
cdef xorshift32(unsigned int*  state):
    """
    Small and good quality PRNG used for lzjd_fSH variant. State is a single 32 bit word
    """
    #Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
    cdef unsigned int x = state[0];
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state[0] = x;
    return x;


@cython.boundscheck(False)
@cython.wraparound(False)   # Deactivate negative indexing.
def lzjd_fSH(const unsigned char[:] input_bytes, unsigned int hash_size):
    """
    This method computes the LZJD set using NEW
    """
    cdef set s1 = computeLZset(input_bytes, hash_size)
    
    cdef unsigned int pos = 0
    cdef unsigned int i
    cdef signed int v
    cdef unsigned int setLength = len(s1)
    
    cdef np.ndarray[float, ndim=1, mode="c"] h = np.full(shape=(hash_size), fill_value=2**32, dtype=np.float32) #use 2^32 instead of inf b/c min() call later will error otherwise
    
    #Copy set into a new dense array, and intialize helper arrays for SuperMinHash algo
    cdef unsigned signed int* d = <signed int *>malloc(setLength * cython.sizeof(int))
    cdef signed int* q = <signed int *>malloc(hash_size * cython.sizeof(int))
    cdef signed int* b = <signed int *>malloc(hash_size * cython.sizeof(int))
    cdef signed int* p = <signed int *>malloc(hash_size * cython.sizeof(int))
    for v in s1:
        d[pos] = v
        pos = pos + 1
        
    for i in range(hash_size):
        q[i] = -1
        b[i] = 0
    b[hash_size-1] = hash_size
    
    
    
    
    cdef unsigned int n = setLength
    cdef unsigned int m = hash_size
    cdef unsigned int a = m - 1
    cdef unsigned int PRNG_state
    cdef unsigned int j
    
    cdef unsigned int r_int
    cdef unsigned long k_tmp
    cdef unsigned int k
    cdef unsigned int swap_tmp 
    cdef float r
    cdef unsigned int jp
    
    for i in range(n):
        #initialize pseudo-random generator with seed d_i
        #Lets use a large prime times the feature hash value
        PRNG_state = max(1, d[i])
        j = 0
        while j <= a:
            #r ← uniform random number from [0, 1) 
            r_int = xorshift32(&PRNG_state)
            r_int >>= 9 # We only want 24 bits of randomness, b/c we need to divide by a value that will fit well as a float
            r = r_int / float(1 << 24)
            #k ← uniform random number from {j, . . . ,m − 1}
            #Ugly way for now, lets just store our rand value in a bigger store (long), then divide.
            #as if we got a value in [0, 1) and then multiplied by what we needed
            k_tmp = <unsigned long>xorshift32(&PRNG_state)
            k_tmp *= (m-1-j)
            k_tmp /= <long>(0xffffffff) #Max int value
            k = <int>k_tmp
            k += j
            
            if q[j] != i:
                #q_j ← i,  p_j ← j
                q[j] = i
                p[j] = j
            #end if
            if q[k] != i:
                #q_k ← i, p_k ← k
                q[k] = i
                p[k] = k
            #swap p_j and p_k
            swap_tmp = p[j]
            p[j] = p[k]
            p[k] = swap_tmp
            
            if r + j < h[p[j]]:
                jp = min(floor(h[p[j]]), m-1)
                h[p[j]] = r + j
                if j < jp:
                    b[jp] -= 1
                    b[j] += 1
                    while b[a] == 0:
                        a -= 1
                    #end while
                #end if
            #end if
            j += 1
    free(d)
    free(q)
    free(b)
    
    #Can the return type be int*?
    return h, setLength
    
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
    if k == pivotIndex or (right - left) <= 1:
        return
    elif k < pivotIndex:
        q_select(arr, left, pivotIndex, k)
    else:
        q_select(arr, pivotIndex + 1, right, k)

#arr: array of elements
#left: the left most position of the array (inclusive)
#right: the right most position of the array (exclusive)
#returns position of the pivot
cdef int q_partition(int* arr, int left, int right):
    cdef int pivot = arr[left]
    cdef int temp 
    cdef int i = left - 1
    cdef int j = right
    while True:
        #Find leftmost element greater than or equal to pivot 
        i = i + 1
        while arr[i] < pivot:
            i = i + 1
        #Find rightmost element smaller than  or equal to pivot 
        j = j - 1
        while arr[j] > pivot:
            j = j - 1
   
        if i >= j:
            return j
   
        temp = arr[j]
        arr[j] = arr[i]
        arr[i] = temp
        

    
cdef int compare(const_void *va, const_void *vb):
    cdef int a = (<signed int *>va)[0]
    cdef int b = (<signed int *>vb)[0]
    return (a > b) - (a < b)
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def intersection_size(int[::1] A, int[::1] B):
    cdef unsigned int pos_a = 0
    cdef unsigned int pos_b = 0
    cdef unsigned int a_len = A.shape[0]
    cdef unsigned int b_len = B.shape[0]
    cdef int int_size = 0
    
    while pos_a < a_len and pos_b < b_len:
        if A[pos_a] < B[pos_b]:
            pos_a += 1
        elif A[pos_a] > B[pos_b]:
            pos_b += 1
        else: #Equal, increment both AND counter!
            pos_a += 1
            pos_b += 1
            int_size += 1
    
    return int_size

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def k_bit_float2vec(float[::1] A, unsigned int k ):
    """
    This method takes as input an array of floating point values associated with a SuperMinHash. 
    To apply the k-bit approach of converting a min-hash to a feature vector from 
    "Hashing Algorithms for Large-Scale Learning", we need to access the integer representation of
    the data. 
    
    This method does that work, and returns a feature vector of the appropriate size. 
    -- A is the input min-hash
    -- k is the number of bits to use in converting it to a feature vector
    """
    
    cdef unsigned int out_size = A.shape[0] * (1<<k)
    cdef unsigned int mask = (1<<k)-1 # This is the bit mask to apply to features. 
    
    #Dont use np array, dense wastes too much memory
    #cdef np.ndarray[float, ndim=1, mode="c"] h = np.zeros(shape=(out_size), dtype=np.float32)
    #Well use a scipy sparase array, defined by data and non-zero row position
    #Its a one hot vector, and nnz = A.shape[0], so just fill with 1.0
    cdef np.ndarray[float, ndim=1, mode="c"] data = np.full(shape=(A.shape[0]), fill_value=1.0, dtype=np.float32)
    #row index is easy b/c we are doing a single row, so its all the same row
    cdef np.ndarray[int, ndim=1, mode="c"] row_ind = np.full(shape=(A.shape[0]), fill_value=0, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode="c"] col_ind = np.zeros(shape=(A.shape[0]), dtype=np.int32)
    
    cdef unsigned int i 
    cdef unsigned int raw_bytes
    cdef unsigned int pos
    
    
    cdef unsigned int* raw_data = <unsigned int*> &A[0]
    
    for i in range(A.shape[0]):
        #raw_bytes = raw_data[i]
        raw_bytes = (<unsigned int*>&(A[i]))[0]
        #Lets apply some iters of XORshift to get better distribution, b/c we might get weirdness with layout of bits in floats for mantisa and exponent 
        xorshift32(&raw_bytes)
        xorshift32(&raw_bytes)
        pos = (1<<k)*i
        pos += raw_bytes & mask
        #h[pos] = 1.0
        col_ind[i] = pos
    
    #return np.zeros(shape=(1), dtype=np.int32)
    #return (data, (row_ind, col_ind))
    return csr_matrix((data, (row_ind, col_ind)), shape=(1,out_size))