import theano
import sys
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
from theano_toolkit import updates
import cPickle       as pickle

from numpy_hinton import print_arr
from theano.printing import Print
from vocab import read_file


def create_vocab_vectors(vocab2id,size):
	V   = U.create_shared(U.initial_weights(len(vocab2id) + 1,size))
	V_b = U.create_shared(U.initial_weights(len(vocab2id) + 1))
	return V,V_b


def recurrent_combine(state_0,X,W_input,W_state,b_state):
	def step(curr_input,state_p):
		# Build next layer
		state = T.dot(curr_input,W_input) + T.dot(state_p,W_state) + b_state
		state = T.tanh(state)

		return state

	states,_ = theano.scan(
			step,
			sequences    = [X],
			outputs_info = [state_0]
		)

	return states

def word_cost(probs,Y):
	return -T.mean(T.log(probs[:-1,Y[1:]]))
#	return -T.mean(T.log(probs[:,Y]))

def create_model(ids,vocab2id,size):
	word_vector_size = size
	rae_state_size   = 2*size
	predictive_hidden_size = rae_state_size 

	V,b_predict = create_vocab_vectors(vocab2id,word_vector_size)
	W_predict = U.create_shared(U.initial_weights(predictive_hidden_size,V.get_value().shape[0]))
	X = V[ids]
	
	# RAE parameters
	W_input = U.create_shared(U.initial_weights(word_vector_size,rae_state_size))
	W_state = U.create_shared(U.initial_weights(rae_state_size,rae_state_size))
	b_state = U.create_shared(U.initial_weights(rae_state_size))
	state_0 = U.create_shared(U.initial_weights(rae_state_size))

	W_state_p_hidden = U.create_shared(U.initial_weights(rae_state_size,predictive_hidden_size))
	b_hidden         = U.create_shared(U.initial_weights(predictive_hidden_size))

	
	states = recurrent_combine(state_0,X,W_input,W_state,b_state)
	

#	hidden = T.dot(states,W_state_p_hidden) + b_hidden
#	hidden = T.tanh(hidden)
#	hidden = hidden * (hidden > 0)
	
	scores = T.dot(states,W_predict) + b_predict
	scores = T.nnet.softmax(scores)

	parameters = [
			V,
			W_input,
			W_state,
			b_state,
			state_0,
#			W_state_p_hidden,
#			b_hidden,
#			W_predict,
			b_predict
		]

	cost = word_cost(scores,ids) # + 1e-5 * sum( T.sum(w**2) for w in parameters )
	return scores, cost, parameters

def training_model(vocab2id,size):
	ids = T.ivector('ids')
	scores, cost, parameters = create_model(ids,vocab2id,size)


	gradients = T.grad(cost,wrt=parameters)
	print "Computed gradients"

	train = theano.function(
			inputs  = [ids],
			updates = updates.adadelta(parameters,gradients,0.95,1e-6),
			outputs = cost
		)

	test = theano.function(
			inputs  = [ids],
			outputs = T.argmax(scores)
		)

	return test,train,parameters


if __name__ == "__main__":
	from train import sentences

	vocab_file = sys.argv[1]
	sentence_file = sys.argv[2]
	vocab2id = pickle.load(open(vocab_file,'r'))
	test,train, parameters = training_model(vocab2id,50)
	for s in sentences(vocab2id,sentence_file):
		s = np.array(s,dtype=np.int32)
		score = train(s)
		print score


