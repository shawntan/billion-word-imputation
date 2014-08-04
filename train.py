# coding=utf-8
import theano
import sys
import theano.tensor as T
import numpy         as np
import utils         as U
import cPickle       as pickle
import random
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

test,train, parameters = model.training_model(vocab2id,100)

#print test(np.array([1,3,4,5,6],dtype=np.int32))
lr = 0.0001
t = 0
for s in sentences(vocab2id,sentence_file):
	if len(s) <= 5: continue
#	print len(s)
#	print 2,len(s)-3
	removed_idx = random.randint(2,len(s)-3)

#	print s,removed_idx
	s.pop(removed_idx)
#	print s
	s = np.array(s,dtype=np.int32)
	score = train(s,removed_idx-1,lr,min(1 - 3.0/(t+5),0.999))
	print score
	if t%100 == 0: 
		print
		print test(s),removed_idx-1
		print
	t += 1


