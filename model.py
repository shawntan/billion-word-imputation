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

def create_vocab_vectors(vocab2id,size):
	V = U.create_shared(U.initial_weights(len(vocab2id) + 1,size))
	return V

def pair_combine(X,W1,W2,b):

	def step(i,inputs):
		length = inputs.shape[0]

		next_level = T.dot(inputs[T.arange(0,length-i-1)],W1) + T.dot(inputs[T.arange(1,length-i)],W2) + b
		#next_level = next_level*(next_level > 0)
		next_level = T.tanh(next_level) 

		return T.concatenate([next_level,T.zeros_like(inputs[:length-next_level.shape[0]])])

	combined,_ = theano.scan(
			step,
			sequences    = [T.arange(X.shape[0])],
			outputs_info = [X],
			n_steps = X.shape[0]-1
		)

	return combined[-1,0], combined[0][:-1]


def cost(score,Y):
	return -T.sum(score - score[Y] + 1)
	
def create_model(vocab2id,size):
	V  = create_vocab_vectors(vocab2id,size)
	ids = T.ivector('ids')
	X  = V[ids]

	W1 = U.create_shared(U.initial_weights(size,size))
	W2 = U.create_shared(U.initial_weights(size,size))
	b  = U.create_shared(U.initial_weights(size))

	W_pairwise = U.create_shared(U.initial_weights(size))
	W_context  = U.create_shared(U.initial_weights(size))
	b_output   = U.create_shared(U.initial_weights(1))

	context, pairwise = pair_combine(X,W1,W2,b)

	score = T.dot(context,W_context) + T.dot(pairwise,W_pairwise) + b_output[0]
	
	f = theano.function(
			inputs = [ids],
			outputs = score
		)
	return f

