# coding: utf-8

import os
import sys
import glob
import math
import gzip
import random
import string
import itertools
import subprocess
import scipy.stats
import scipy.optimize
import scipy.spatial.distance
from multiprocessing import Pool
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors


def load_corpus():
    gold_standards = list()
    sentence_pairs = list()
    sentences = list()
    vocab = set()
    for fin_name, fgs_name in zip(sorted(glob.glob("test_evaluation_task2a/*.input.*.txt")), sorted(glob.glob("test_evaluation_task2a/*.gs.*.txt"))):
        fin = open(fin_name, "r")
        fgs = open(fgs_name, "r")
        for line_in, line_gs in zip(fin, fgs):
            if not line_gs.strip():
                continue
            gold_standards.append(float(line_gs.strip()))
            sentence1, sentence2 = [tokenize(sentence) for sentence in line_in.strip().split("\t")]
            sentence_pairs.append((sentence1, sentence2))
            sentences.append(sentence1)
            sentences.append(sentence2)
            vocab |= (set(sentence1) | set(sentence2))
        fin.close()
        fgs.close()
    return gold_standards, sentence_pairs, sentences, vocab


def tokenize(sentence):
    return "".join([character for character in sentence.lower() if character not in string.punctuation]).split()


def original_similarity(fname_similarity, corpus_vocab):
    w2v_model, w2v_vocab = load_word2vec()
    vocab = list(w2v_vocab & corpus_vocab)
    fout = gzip.open(fname_similarity, "w")
    for word1, word2 in list(itertools.product(vocab, repeat=2)):
        fout.write("\t".join([word1, word2, "%1.3f" % max(0.0, w2v_model.similarity(word1, word2))]) + "\n")
    fout.close()


def load_word2vec():
    w2v_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    vocab = set(w2v_model.vocab.keys())
    return w2v_model, vocab


def load_similarity(fname):
    word_similarity = defaultdict(lambda: defaultdict(float))
    fin = gzip.open(fname, "r")
    for line in fin:
        word1, word2, similarity = line.strip().split("\t")
        word_similarity[word1][word2] = float(similarity)
    fin.close()
    return word_similarity, word_similarity.keys()


def load_alignment(fname):
    word_alignment = defaultdict(lambda: defaultdict(float))
    fin = gzip.open(fname, "r")
    for line in fin:
        word1, word2, freq = line.strip().split("\t")
        word_alignment[word1][word2] = float(freq)
    fin.close()
    return ppmi(word_alignment, word_alignment.keys())


def ppmi(word_alignment, vocab):
    sum_freq = int()
    sum_i = defaultdict(float)
    sum_j = defaultdict(float)
    for word1, word2 in list(itertools.product(vocab, repeat=2)):
        freq = (word_alignment[word1][word2] + word_alignment[word2][word1]) / 2.0
        sum_freq += freq
        sum_i[word1] += freq
        sum_j[word2] += freq
    alignment_similarity = defaultdict(lambda: defaultdict(float))
    for word1, word2 in list(itertools.product(vocab, repeat=2)):
        freq = float(word_alignment[word1][word2] + word_alignment[word2][word1]) / 2.0
        if freq > 0.0 and sum_i[word1] > 0.0 and sum_j[word2] > 0.0:
            alignment_similarity[word1][word2] = max((0.0, math.log((freq * sum_freq) / (sum_i[word1] * sum_j[word2]))))
        else:
            alignment_similarity[word1][word2] = 0.0
    max_sim = max([alignment_similarity[word1][word2] for word1, word2 in list(itertools.product(vocab, repeat=2))])
    for word1, word2 in list(itertools.product(vocab, repeat=2)):
        alignment_similarity[word1][word2] = alignment_similarity[word1][word2] / max_sim
    return alignment_similarity


def alignment(sentences, t):
    pool = Pool()
    n = 750000
    sentence_combinations = list(itertools.combinations(sentences, 2))
    method = sys.argv[1]
    if method.startswith("Amax"):
        pool.map(maximum_alignment, [sentence_combinations[x:x+n] for x in range(0, len(sentence_combinations), n)])
    elif method.startswith("Ahun"):
        pool.map(hungarian_alignment, [sentence_combinations[x:x+n] for x in range(0, len(sentence_combinations), n)])
    word_alignment = defaultdict(lambda: defaultdict(float))
    for fname in glob.glob("tmp-*.gz"):
        fin = gzip.open(fname, "r")
        for line in fin:
            word1, word2 = line.strip().split("\t")
            word_alignment[word1][word2] += 1.0
        fin.close()
    fout = gzip.open("wa-%s-%s-t%02d.tsv.gz" % (sys.argv[2].replace(".", ""), sys.argv[1], t), "w")
    for word1, word2 in list(itertools.product(word_alignment.keys(), repeat=2)):
        fout.write("\t".join([word1, word2, str(word_alignment[word1][word2])]) + "\n")
    fout.close()
    subprocess.call("rm tmp-*.gz", shell=True)
    return ppmi(word_alignment, word_alignment.keys())


def maximum_alignment(sentence_combinations):
    threshold = float(sys.argv[2])
    try:
        fname_similarity = sorted(glob.glob("ws-%s-%s-t*.tsv.gz" % (sys.argv[2].replace(".", ""), sys.argv[1])))[-1]
    except:
        fname_similarity = "ws-t00.tsv.gz"
    word_similarity, vocab = load_similarity(fname_similarity)
    fout = gzip.open("tmp-%05d.gz" % random.randint(0, 99999), "w")
    for sentence1, sentence2 in sentence_combinations:
        for word1 in sentence1:
            similarity, word2 = max([(word_similarity[word1][word2], word2) for word2 in sentence2])
            if similarity > threshold:
                fout.write(word1 + "\t" + word2 + "\n")
        for word2 in sentence2:
            similarity, word1 = max([(word_similarity[word1][word2], word1) for word1 in sentence1])
            if similarity > threshold:
                fout.write(word1 + "\t" + word2 + "\n")
    fout.close()


def hungarian_alignment(sentence_combinations):
    threshold = float(sys.argv[2])
    try:
        fname_similarity = sorted(glob.glob("ws-%s-%s-t*.tsv.gz" % (sys.argv[2].replace(".", ""), sys.argv[1])))[-1]
    except:
        fname_similarity = "ws-t00.tsv.gz"
    word_similarity, vocab = load_similarity(fname_similarity)
    fout = gzip.open("tmp-%05d.gz" % random.randint(0, 99999), "w")
    for sentence1, sentence2 in sentence_combinations:
        cost_matrix = [[1.0 - word_similarity[word1][word2] for word2 in sentence2] for word1 in sentence1]
        sentence1_ids, sentence2_ids = scipy.optimize.linear_sum_assignment(cost_matrix)
        for sentence1_id, sentence2_id in zip(sentence1_ids, sentence2_ids):
            if word_similarity[sentence1[sentence1_id]][sentence2[sentence2_id]] > threshold:
                fout.write(sentence1[sentence1_id] + "\t" + sentence2[sentence2_id] + "\n")
    fout.close()


def update(word_similarity, alignment_similarity, t):
    method = sys.argv[1]
    fout_name = "ws-%s-%s-t%02d.tsv.gz" % (sys.argv[2].replace(".", ""), sys.argv[1], t + 1)
    if "Uadd" in method:
        return update_add(fout_name, word_similarity, alignment_similarity)
    elif "Umul" in method:
        return update_mul(fout_name, word_similarity, alignment_similarity)


def update_add(fout_name, word_similarity, alignment_similarity):
    fout = gzip.open(fout_name, "w")
    for word1, word2 in list(itertools.product(word_similarity.keys(), repeat=2)):
        word_similarity[word1][word2] = (word_similarity[word1][word2] + alignment_similarity[word1][word2]) / 2.0
        fout.write("\t".join([word1, word2, "%1.3f" % word_similarity[word1][word2]]) + "\n")
    fout.close()
    return word_similarity


def update_mul(fout_name, word_similarity, alignment_similarity):
    fout = gzip.open(fout_name, "w")
    for word1, word2 in list(itertools.product(word_similarity.keys(), repeat=2)):
        word_similarity[word1][word2] = word_similarity[word1][word2] * alignment_similarity[word1][word2]
        fout.write("\t".join([word1, word2, "%1.3f" % word_similarity[word1][word2]]) + "\n")
    fout.close()
    return word_similarity


def sts(sentence_pairs, word_similarity, threshold):
    method = sys.argv[1]
    system_outputs = list()
    for sentence1, sentence2 in sentence_pairs:
        if method.endswith("Savg"):
            similarities = [word_similarity[word1][word2] for word1 in sentence1 for word2 in sentence2]
        elif method.endswith("Smax"):
            similarities = sts_maximum(sentence1, sentence2, word_similarity)
        elif method.endswith("Shun"):
            similarities = sts_hungarian(sentence1, sentence2, word_similarity)
        similarities = [similarity if similarity > threshold else 0.0 for similarity in similarities]
        if similarities:
            system_outputs.append(sum(similarities) / len(similarities))
        else:
            system_outputs.append(0.0)
    return system_outputs


def sts_maximum(sentence1, sentence2, word_similarity):
    forward_similarities = [max([word_similarity[word1][word2] for word2 in sentence2]) for word1 in sentence1]
    backward_similarities = [max([word_similarity[word1][word2] for word1 in sentence1]) for word2 in sentence2]
    return forward_similarities + backward_similarities


def sts_hungarian(sentence1, sentence2, word_similarity):
        cost_matrix = [[1.0 - word_similarity[word1][word2] for word2 in sentence2] for word1 in sentence1]
        sentence1_ids, sentence2_ids = scipy.optimize.linear_sum_assignment(cost_matrix)
        return [word_similarity[sentence1[sentence1_id]][sentence2[sentence2_id]] for sentence1_id, sentence2_id in zip(sentence1_ids, sentence2_ids)]


def evaluate(gold_standards, system_outputs, t, method, threshold):
    pearsonr, p_value = scipy.stats.pearsonr(gold_standards, system_outputs)
    print "t=%02d %s(%1.2f): %1.3f" % (t, method, threshold, pearsonr)


def main():
    if not len(sys.argv) == 3:
        print "usage: python ia4gess.py method threshold"
        print "\t\tmethod: AmaxUaddShun, AhunUmulSmax, ...(A{max,hun}U{add,mul}S{avg,max,hun})"
        print "\t\tthreshold: float ([0.00, 0.99])"
    method = sys.argv[1]
    threshold = float(sys.argv[2])
    gold_standards, sentence_pairs, sentences, vocab = load_corpus()
    t = 0
    while True:
        if not os.path.exists("ws-%s-%s-t%02d.tsv.gz" % (sys.argv[2].replace(".", ""), sys.argv[1], t + 1)):
            break
        t += 1
    if t == 0:
        fname_similarity = "ws-t00.tsv.gz"
        if not os.path.exists(fname_similarity):
            original_similarity(fname_similarity, vocab)
    else:
        fname_similarity = "ws-%s-%s-t%02d.tsv.gz" % (sys.argv[2].replace(".", ""), sys.argv[1], t)
    word_similarity, vocab = load_similarity(fname_similarity)
    evaluate(gold_standards, sts(sentence_pairs, word_similarity, threshold), t, method, threshold)
    alignment_similarity = alignment(sentences, t)
    word_similarity = update(word_similarity, alignment_similarity, t)
    evaluate(gold_standards, sts(sentence_pairs, word_similarity, threshold), t + 1, method, threshold)


if __name__ == '__main__':
    main()
