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
			sequences    = [T.arange(X.shape[0]-1)],
			outputs_info = [X],
		)

	return combined[-1,0], combined[0][:-1]


def recurrent_combine(X,W1,b1,W2,b2,b):
	def step(curr_in,hidden):
		next_level = T.dot(curr_in,W1) + T.dot(hidden,W2) + b
		next_level = T.tanh(next_level)
		
		reproduction_curr_in = T.dot(next_level,W1.T) + b1
		reproduction_hidden  = T.dot(next_level,W2.T) + b2
		return next_level,reproduction_curr_in,reproduction_hidden

	[hiddens,rep_ins,rep_hiddens],_ = theano.scan(
			step,
			sequences    = [X[1:]],
			outputs_info = [X[0],None,None]
		)

	return hiddens, rep_ins, rep_hiddens



def missing_cost(scores,Y):
	label_score = scores[Y]
	scores_diff = -(label_score - (scores + 1))
	scores_diff = scores_diff * (scores_diff > 0)
	total_scores_diff = (T.sum(scores_diff) - scores_diff[Y])/(scores.shape[0]-1)
	return total_scores_diff

def rae_cost(X,hiddens,rec_ins,rec_hiddens):
	input_rec_cost = T.mean(T.sum((X[1:]-rec_ins)**2,axis=1))
#	hiddens = T.concatenate([X[0],hiddens[:-1]])
	hidden_rec_cost = (T.sum((hiddens[:-1] - rec_hiddens[1:])**2) + T.sum((X[0] - rec_hiddens[0])**2))/hiddens.shape[0]
	return input_rec_cost + hidden_rec_cost
	

	
def create_model(ids,Y,vocab2id,size):
	V   = create_vocab_vectors(vocab2id,size)
	X   = V[ids]

	W1 = U.create_shared(U.initial_weights(size,size))
	b1 = U.create_shared(U.initial_weights(size))
	W2 = U.create_shared(U.initial_weights(size,size))
	b2 = U.create_shared(U.initial_weights(size))
	b  = U.create_shared(U.initial_weights(size))

	hiddens, rec_ins, rec_hiddens = recurrent_combine(X,W1,b1,W2,b2,b)
	context = hiddens[-1]
	pairwise = hiddens

	W_pairwise = U.create_shared(U.initial_weights(size,size))
	W_context  = U.create_shared(U.initial_weights(size,size))
	b_hidden   = U.create_shared(U.initial_weights(size))

	hidden = T.dot(context,W_context) + T.dot(pairwise,W_pairwise) + b_hidden
	hidden = T.tanh(hidden)
#	hidden = hidden * (hidden > 0)
	
	W_output = U.create_shared(U.initial_weights(size))
	scores = T.dot(hidden,W_output)

	#parameters = [V,W1,W2,b,W_pairwise,W_context,b_output]
	parameters = [V,W1,b1,W2,b2,b,W_pairwise,W_context,b_hidden,W_output]

	cost = rae_cost(X,hiddens,rec_ins,rec_hiddens) + missing_cost(scores,Y) + 1e-5*sum(T.sum(w**2) for w in parameters)
	return scores, cost, parameters

def training_model(vocab2id,size):
	ids = T.ivector('ids')
	Y = T.iscalar('Y')
	scores, cost, parameters = create_model(ids,Y,vocab2id,size)


	gradients = T.grad(cost,wrt=parameters)
	print "Computed gradients"

	eps = T.dscalar('eps')
	mu  = T.dscalar('mu')
	deltas = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	delta_nexts = [ mu*delta + eps*grad for delta,grad in zip(deltas,gradients) ]
	delta_updates = [ (delta, delta_next) for delta,delta_next in zip(deltas,delta_nexts) ]
	param_updates = [ (param, param - delta_next) for param,delta_next in zip(parameters,delta_nexts) ]

	train = theano.function(
			inputs  = [ids,Y,eps,mu],
			updates = delta_updates + param_updates,
			outputs = cost
		)

	test = theano.function(
			inputs  = [ids],
			outputs = T.argmax(scores)
		)

	return test,train, parameters
