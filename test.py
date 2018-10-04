import pyximport; pyximport.install()
import lzjd_cython


bytes = str.encode("Heel"*20)
d = lzjd_cython.lzjd_f(bytes, 1024)
print(len(d))
print(d)