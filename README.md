# pyLZJD

pyLZJD is a Python implementatio of the *Lempel-Ziv Jaccard Distance*, a distance metric designed for arbitrary byte sequences, and originally used for malware classification. It was inspired by and developed as an alternative to 
the 
[Normalized Compression Distance](https://en.wikipedia.org/wiki/Normalized_compression_distance). But, we've also found it useful for similarity digest taks, where one would normaly use either 
[ssdeep](http://www.forensicswiki.org/wiki/Ssdeep) or [sdhash](http://roussev.net/sdhash/tutorial/03-quick.html). 

## Why use pyLZJD? 

If you need to find similar byte sequences, and your byte sequences can be long (>500kb), then you should consider using LZJD and this implementation! LZJD is fast, efficient, and we've found it to be more accurate than the 
previously existing options. If you want to know more nity gritty details, check out the two papers listed under citations. 

We currently recommend the pyLZJD implementation for ptototyipng and learning / experimenting with the LZJD algorithm. This code is not yet production ready, and better and faster implementations exist.

## Why not use pyLZJD?

There currently exists the original [Java implementation](https://github.com/EdwardRaff/jLZJD) and a [C++ implementation](https://github.com/EdwardRaff/LZJD) of the LZJD algorithm. While we have implemented the main portion in Cython, this version is currently 3-20 times slower than these more optimized implementations. If you need efficiency or plan to report timing numbers, please do not use this version. 

# Insallation 

To install pyLZJD, you can currently use this syntax with pip:
```
pip install git+git://github.com/EdwardRaff/pyLZJD#egg=pyLZJD
```
 
 Or, you can download the repo and run
```
python setup.py install
```

## Citations

If you use LZJD, please cite it! There are currently two papers related to LZJD. The [original paper](http://www.edwardraff.com/publications/alternative-ncd-lzjd.pdf) that introduces it, and a [followup paper](https://arxiv.org/abs/1708.03346) that shows how LZJD 
can be used inplace of ssdeep and sdhash, and makes LZJD even faster. Please cite the first paper if you use LZJD at all, and please cite the second as well if you use this implementation as it uses the faster version introduced in 
the second paper. 

```
@inproceedings{raff_lzjd_2017,
 author = {Raff, Edward and Nicholas, Charles},
 title = {An Alternative to NCD for Large Sequences, Lempel-Ziv Jaccard Distance},
 booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
 series = {KDD '17},
 year = {2017},
 isbn = {978-1-4503-4887-4},
 location = {Halifax, NS, Canada},
 pages = {1007--1015},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3097983.3098111},
 doi = {10.1145/3097983.3098111},
 acmid = {3098111},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {cyber security, jaccard similarity, lempel-ziv, malware classification, normalized compression distance},
}

@article{raff_lzjd_digest,
author = {Raff, Edward and Nicholas, Charles K.},
doi = {10.1016/j.diin.2017.12.004},
issn = {17422876},
journal = {Digital Investigation},
month = {feb},
title = {{Lempel-Ziv Jaccard Distance, an effective alternative to ssdeep and sdhash}},
url = {https://doi.org/10.1016/j.diin.2017.12.004},
year = {2018}
}

```
