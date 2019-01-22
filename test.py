from pyLZJD import digest, sim


byte_data = str.encode("Heel"*2000)
print(type(byte_data))
d = digest(byte_data)
#print(len(d))
#print(d)

print(type(d))
print( sim(d, d))

print(sim(d, digest("Heel"*1000 + "Boy"*1000)))

f_a = "lzjd.py"
f_b = "lzjd_cython.pyx"

print(sim(digest(f_a), digest(f_b)))


hashes = digest([f_a, f_b]*1000, processes=-1)
print(sim(hashes[0], hashes[1]))