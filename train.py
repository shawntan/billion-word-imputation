# coding=utf-8
import theano
import sys
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
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

if __name__ == "__main__":
	vocab_file = sys.argv[1]
	sentence_file = sys.argv[2]
	vocab2id = pickle.load(open(vocab_file,'r'))

	id2vocab = [None]*len(vocab2id)
	for k,v in vocab2id.iteritems(): id2vocab[v]=k

	test,train, parameters = model.training_model(vocab2id,50)

	#print test(np.array([1,3,4,5,6],dtype=np.int32))
	lr = 0.0001
	t = 0
	for s in sentences(vocab2id,sentence_file):
		if len(s) <= 5: continue
		choices = [
			i for i in xrange(len(s)) 
				if  len(id2vocab[s[i]]) > 2 and\
					i > 1 and i < (len(s) - 2)
			]
		if len(choices) == 0: continue
		removed_idx = random.choice(choices)
	#	print s,removed_idx
		missing = s.pop(removed_idx)
	#	print s
		s = np.array(s,dtype=np.int32)
		if t%100 == 0:
			words = [id2vocab[idx] if idx != -1 else 'UNK' for idx in s]
			pred_pos = test(s) + 1
			act_pos  = removed_idx
			if pred_pos == act_pos:
				words.insert(act_pos, "~%s~"%id2vocab[missing])
			else:
				if act_pos > pred_pos: act_pos += 1
				words.insert(pred_pos,"^^^^^")
				words.insert(act_pos, "|%s|"%id2vocab[missing])

			print
			print ' '.join(words)
			print
		else:
			score = train(s,removed_idx-1)
			print score

		t += 1


