# coding=utf-8
import theano
import sys
import theano.tensor as T
import numpy         as np
import utils         as U
import cPickle       as pickle

from numpy_hinton import print_arr
from theano.printing import Print
from vocab import read_file
import model

def sentences(vocab2id,filename):
	for tokens in read_file(filename):
		#print tokens
		for i in xrange(len(tokens)): 
			#if tokens[i] not in vocab2id: print tokens[i]
			tokens[i] = vocab2id.get(tokens[i],-1)
		yield tokens

vocab_file = sys.argv[1]
sentence_file = sys.argv[2]
vocab2id = pickle.load(open(vocab_file,'r'))

f = model.create_model(vocab2id,100)
for s in sentences(vocab2id,sentence_file):
	score = f(np.array(s,dtype=np.int32))
	print len(s), score.shape


