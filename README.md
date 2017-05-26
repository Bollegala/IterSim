# IterSim


 [![DOI](https://zenodo.org/badge/92485470.svg)](https://zenodo.org/badge/latestdoi/92485470)

# An Iterative Approach for the Global Estimation of Sentence Similarity

## Dependencies
- phthon 2.7
- gensim
- word embeddings (e.g. [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec/))
- sentence pairs with similarity score (e.g. [SemEval-2015 corpus](http://alt.qcri.org/semeval2015/task2/data/uploads/test_evaluation_task2a.tgz))

## Usage
```
$ python ia4gess.py method threshold
```
- method
	- AmaxUaddSavg: Average similarity based on additive update with maximum alignment
	- AmaxUaddSmax: Maximum similarity based on additive update with maximum alignment
	- AmaxUaddShun: Hungarian similarity based on additive update with maximum alignment
	- AhunUaddSavg: Average similarity based on additive update with hungarian alignment
	- AhunUaddSmax: Maximum similarity based on additive update with hungarian alignment
	- AhunUaddShun: Hungarian similarity based on additive update with hungarian alignment
	- AmaxUmulSavg: Average similarity based on multiplicative update with maximum alignment
	- AmaxUmulSmax: Maximum similarity based on multiplicative update with maximum alignment
	- AmaxUmulShun: Hungarian similarity based on multiplicative update with maximum alignment
	- AhunUmulSavg: Average similarity based on multiplicative update with hungarian alignment
	- AhunUmulSmax: Maximum similarity based on multiplicative update with hungarian alignment
	- AhunUmulShun: Hungarian similarity based on multiplicative update with hungarian alignment
- threshold
	- [0.00, 0.99]: threshold for word alignment

For questions, please contact [Tomoyuki Kajiwara at Tokyo Metropolitan University](https://sites.google.com/site/moguranosenshi/).
