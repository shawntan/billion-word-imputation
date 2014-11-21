import theano
import sys
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
import cPickle       as pickle
from lang_model import *

np.random.seed()

def build_sampler(vocab2id,size):
	word_vector_size = size
	rae_state_size   = size
	predictive_hidden_size = rae_state_size

	P = Parameters()
	V,b_predict = create_vocab_vectors(P,vocab2id,word_vector_size)
	W_predict = V.T
#	W_predict = U.create_shared(U.initial_weights(predictive_hidden_size,V.get_value().shape[0]))
	
	# RAE parameters
	P.W_transform = U.initial_weights(rae_state_size,rae_state_size)
	P.b_transform = U.initial_weights(rae_state_size)
	P.W_state = U.initial_weights(rae_state_size,rae_state_size)
	P.W_input = U.initial_weights(rae_state_size,rae_state_size)
	P.b_state = U.initial_weights(rae_state_size)
	
	P.state_0 = U.initial_weights(rae_state_size)

	idx = T.iscalar('idx')
	state_p = T.vector('state_p')

	
	X = T.dot(V[idx],P.W_transform) + P.b_transform
	state = T.dot(state_p,P.W_input) * X + T.dot(state_p,P.W_state) + P.b_state
	state = T.tanh(state)
	scores = T.dot(state,W_predict) + b_predict
	scores = T.nnet.softmax(scores)[0]


	sample_next = theano.function(
			inputs = [idx,state_p],
			outputs = [scores,state]
		)
	return sample_next,P

if __name__ == "__main__":
	vocab_file = sys.argv[1]
	params_file = sys.argv[2]
	
	print "Loading vocab..."
	vocab2id = pickle.load(open(vocab_file,'r'))
	id2vocab = [None]*len(vocab2id)
	for k,v in vocab2id.iteritems(): id2vocab[v] = k

	print "Build model..."
	sample_next,P = build_sampler(vocab2id,20)
	print "Loading params..."
	P.load(sys.argv[2])
	state_0 = P.state_0.get_value()
	start_idx = vocab2id['<START>']
	end_idx = vocab2id['<END>']
	for _ in xrange(10):
		probs,state_p = sample_next(start_idx,state_0)
		probs = np.asarray(probs,dtype=np.float64)
		probs = probs/probs.sum()
		word_idx = np.random.choice(len(id2vocab)+1,1,p=probs)[0]
		while word_idx != end_idx:
			print (id2vocab[word_idx] if word_idx < len(id2vocab) else "<unk>"),
			probs,state_p = sample_next(word_idx,state_p)
			probs = np.asarray(probs,dtype=np.float64)
			probs = probs/probs.sum()
			word_idx = np.random.choice(len(id2vocab)+1,1,p=probs)[0]
		print ("\n"),
