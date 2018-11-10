from lzjd import hash, sim


byte_data = str.encode("Heel"*2000)
print(type(byte_data))
d = hash(byte_data)
#print(len(d))
#print(d)

print(type(d))
print( sim(d, d))

print(sim(d, hash("Heel"*1000 + "Boy"*1000)))

f_a = "lzjd.py"
f_b = "lzjd_cython.pyx"

print(sim(hash(f_a), hash(f_b)))