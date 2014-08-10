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
	V   = U.create_shared(U.initial_weights(len(vocab2id) + 1,size))
	V_b = U.create_shared(U.initial_weights(len(vocab2id) + 1))
	return V,V_b


def recurrent_combine(X,V,V_b,W_input,b_input,W_state_p,b_state_p,b_state,W_input_hidden,W_state_p_hidden):
	def step(curr_input,state_p):
		# Build next layer
		state = T.dot(curr_input,W_input) + T.dot(state_p,W_state_p) + b_state
		state = T.tanh(state)

		# RAE stuff
		rep_word_vec   = T.dot(state,W_input.T) + b_input
		rep_curr_input = T.dot(rep_word_vec,V.T) + V_b
		rep_state_p    = T.dot(state,W_state_p.T) + b_state_p

		# Contributions to predictive hidden layer
		hidden_partial = T.dot(state_p,W_state_p_hidden) + T.dot(curr_input,W_input_hidden)

		return state,rep_curr_input,rep_state_p,hidden_partial

	[states,rep_inputs,rep_states,hidden_partials],_ = theano.scan(
			step,
			sequences    = [X[1:]],
			outputs_info = [X[0],None,None,None]
		)

	return states,T.nnet.softmax(rep_inputs),rep_states,hidden_partials


def missing_cost(scores,Y):
	probs = T.nnet.softmax(scores)[0]
	total_scores_diff = -T.log(probs[Y])
	"""
	label_score = scores[Y]
	scores_diff = -(label_score - (scores + 1))
	scores_diff = scores_diff * (scores_diff > 0)
	total_scores_diff = (T.sum(scores_diff) - scores_diff[Y])/(scores.shape[0]-1)
	"""
	return total_scores_diff

def rae_cost(ids,X,states,rep_inputs,rep_states):
	# Actual input - reconstructed input error
	#input_rec_cost = T.mean(T.sum((X[1:]-rep_inputs)**2,axis=1))
	input_rec_cost = -T.mean(T.log(rep_inputs[T.arange(rep_inputs.shape[0]),ids[1:]]))
	# Actual prev state - reconstructed prev state error
	state_rec_cost = (
			# All states except last, all rec states except first
			T.sum((states[:-1] - rep_states[1:])**2) +\
			# First state (first input) and first rec state
			T.sum((X[0] - rep_states[0])**2)
		)/states.shape[0]
	return input_rec_cost + state_rec_cost


def create_model(ids,Y,vocab2id,size):
	word_vector_size = size
	rae_state_size   = size
	predictive_hidden_size = size * 2
	

	V,V_b = create_vocab_vectors(vocab2id,word_vector_size)
	X     = V[ids]
	
	# RAE parameters
	W_input   = U.create_shared(U.initial_weights(word_vector_size,rae_state_size))
	b_input   = U.create_shared(U.initial_weights(rae_state_size))
	W_state_p = U.create_shared(U.initial_weights(rae_state_size,rae_state_size))
	b_state_p = U.create_shared(U.initial_weights(rae_state_size))
	b_state   = U.create_shared(U.initial_weights(rae_state_size))

	W_input_hidden   = U.create_shared(U.initial_weights(word_vector_size,predictive_hidden_size))
	W_state_p_hidden = U.create_shared(U.initial_weights(rae_state_size,predictive_hidden_size))

	W_full_context_hidden = U.create_shared(U.initial_weights(rae_state_size,predictive_hidden_size))
	b_hidden              = U.create_shared(U.initial_weights(predictive_hidden_size))

	W_output              = U.create_shared(U.initial_weights(predictive_hidden_size))
	
	states,rep_inputs,rep_states,hidden_partials = recurrent_combine(
			X,
			V,V_b,
			W_input,b_input,
			W_state_p,b_state_p,b_state,
			W_input_hidden,W_state_p_hidden,
		)

	context = states[-1]
	hidden = T.dot(context,W_full_context_hidden) + hidden_partials + b_hidden
#	hidden = T.tanh(hidden)
	hidden = hidden * (hidden > 0)
	
	scores = T.dot(hidden,W_output)

	parameters = [
			V,
			V_b,
			W_input,
			b_input,
			W_state_p,
			b_state_p,
			b_state,
			W_input_hidden,
			W_state_p_hidden,
			W_full_context_hidden,
			b_hidden,
			W_output
		]

	cost = rae_cost(ids,X,states,rep_inputs,rep_states) + missing_cost(scores,Y) + 1e-5*sum(T.sum(w**2) for w in parameters)
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
