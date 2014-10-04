import theano
import sys
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
import cPickle       as pickle
from lang_model import *
from theano.tensor.shared_randomstreams import RandomStreams
U.theano_rng = RandomStreams()
def recurrent_sample(V,state_0,W_input,W_state,b_state,W_predict,b_predict,start_idx,stop_idx):
	i_vec = T.eye(1)[0]
	def step(curr_input,state_p):
		print curr_input.type
		curr_input = T.cast(curr_input[0],'int32')
		print curr_input.type
		curr_input = V[curr_input]
		state  = T.dot(curr_input,W_input) + T.dot(state_p,W_state) + b_state
		state  = T.tanh(state)
		activations = T.exp(T.dot(state,W_predict) + b_predict)
		probs = activations/T.sum(activations)
	 	select = T.argmax(U.theano_rng.multinomial(pvals=probs,ndim=1))

		select = select*i_vec
		return [select,state],theano.scan_module.until(T.eq(select[0],stop_idx))

	[sequence,_],updates = theano.scan(
			step,
			outputs_info = [start_idx*i_vec,state_0],
			n_steps = 100
		)
	return sequence,updates




def sampling_model(vocab2id,size):
	word_vector_size = size
	rae_state_size   = size
	predictive_hidden_size = rae_state_size

	P = Parameters()
	V,b_predict = create_vocab_vectors(P,vocab2id,word_vector_size)
	W_predict = V.T
#	W_predict = U.create_shared(U.initial_weights(predictive_hidden_size,V.get_value().shape[0]))

	# RAE parameters
	P.W_state = U.initial_weights(rae_state_size,rae_state_size)
	P.W_input = U.initial_weights(rae_state_size,rae_state_size)
	P.b_state = U.initial_weights(rae_state_size)
	P.state_0 = U.initial_weights(rae_state_size)

	sequence,updates = recurrent_sample(V,P.state_0,P.W_input,P.W_state,P.b_state,W_predict,b_predict,vocab2id['<START>'],vocab2id['<END>'])
	generate = theano.function(
			inputs=[],
			outputs=sequence,
			updates=updates
		)
	return generate,P


if __name__ == "__main__":
	vocab_file = sys.argv[1]
	params_file = sys.argv[2]
	
	print "Loading vocab..."
	vocab2id = pickle.load(open(vocab_file,'r'))
	id2vocab = [None]*len(vocab2id)
	for k,v in vocab2id.iteritems(): id2vocab[v] = k
	print "Build model..."
	generate,P = sampling_model(vocab2id,96)
	print "Loading params..."
	P.load(sys.argv[2])
	for _ in xrange(10):
		sequence = generate()
		sequence = [ int(idx[0]) for idx in sequence ][:-1]
		print ' '.join(id2vocab[idx] if idx < len(id2vocab) else "<unk>" for idx in sequence)

