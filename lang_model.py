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
#		state = curr_input + T.dot(state_p,W_state) + b_state
		state = T.dot(curr_input,W_input) + T.dot(state_p,W_state) + b_state
#		state = T.nnet.sigmoid(state)
		state = T.tanh(state)

		return state

	states,_ = theano.scan(
			step,
			sequences    = [X],
			outputs_info = [state_0]
		)

	return states

def word_cost(probs,Y):
	lbl_probs = probs[T.arange(Y.shape[0]),Y]
	return -T.sum(T.log(lbl_probs)), -T.mean(T.log2(lbl_probs))

def rae(W_input,W_state,inputs,state_0,states):
	b_input_rec = U.create_shared(U.initial_weights(W_input.get_value().shape[1]))
	b_state_rec = U.create_shared(U.initial_weights(W_state.get_value().shape[1]))

	input_rec = T.dot(states,W_input.T) + b_input_rec
	state_rec = T.tanh(T.dot(states,W_state.T) + b_state_rec)

	input_rec_cost = T.sum((input_rec - inputs)**2)
	state_rec_cost = T.sum((state_rec[1:] - states[:-1])**2)

	cost = input_rec_cost + state_rec_cost

	return [b_input_rec,b_state_rec],cost





def create_model(ids,vocab2id,size):
	word_vector_size = size
	rae_state_size   = size
	predictive_hidden_size = rae_state_size

	V,b_predict = create_vocab_vectors(vocab2id,word_vector_size)
	W_predict = V.T
#	W_predict = U.create_shared(U.initial_weights(predictive_hidden_size,V.get_value().shape[0]))
	X = V[ids]
	
	# RAE parameters
	W_state = U.create_shared(U.initial_weights(rae_state_size,rae_state_size))
	W_input = U.create_shared(U.initial_weights(rae_state_size,rae_state_size))
	b_state = U.create_shared(U.initial_weights(rae_state_size))
	state_0 = U.create_shared(U.initial_weights(rae_state_size))
	
	states = recurrent_combine(state_0,X,W_input,W_state,b_state)

	scores = T.dot(states,W_predict) + b_predict
	scores = T.nnet.softmax(scores)

	log_likelihood, cross_ent = word_cost(scores[:-1],ids[1:])
	recon_b, rae_cost = rae(W_input,W_state,X,state_0,states)


	parameters = [
			V,
			W_state,
			W_input,
			b_state,
			state_0,
#			W_predict,
			b_predict
		] #+ recon_b


	cost = log_likelihood + 1e-5 * sum( T.sum(w**2) for w in parameters )
	obv_cost = cross_ent
	return scores, cost, obv_cost, parameters

def training_model(vocab2id,size):
	ids = T.ivector('ids')
	scores, cost, obv_cost, parameters = create_model(ids,vocab2id,size)


	gradients = T.grad(cost,wrt=parameters)
	print "Computed gradients"
	train = theano.function(
			inputs  = [ids],
			updates = updates.adadelta(parameters,gradients,rho=0.95,eps=1e-6),
			outputs = obv_cost
		)

	test = theano.function(
			inputs  = [ids],
			outputs = obv_cost
		)

	predict = theano.function(
			inputs  = [ids],
			outputs = T.argmax(scores,axis=1)
		)

	return predict,train,test,parameters

def run_test(vocab2id,test_file,test):
	total,count = 0,0
	for s in sentences(vocab2id,test_file):
		s = np.array(s,dtype=np.int32)
		score = test(s)
		length = len(s) - 1
		total += score * length
		count += length
	return total/count

if __name__ == "__main__":
	from train import sentences

	vocab_file = sys.argv[1]
	sentence_file = sys.argv[2]
	test_file = sys.argv[3]

	vocab2id = pickle.load(open(vocab_file,'r'))
	id2vocab = [None]*len(vocab2id)
	for k,v in vocab2id.iteritems(): id2vocab[v]=k

	predict, train, test, parameters = training_model(vocab2id,20)
	print "Loading params..."
	try:
		saved_params = pickle.load(open('params','rb'))
		for p,sp in zip(parameters,saved_params): p.set_value(sp)
	except:
		pass

	max_test = np.inf
	for epoch in range(10):
		count = 0
		for s in sentences(vocab2id,sentence_file):
			s = np.array(s,dtype=np.int32)
			score = train(s)
			print score
			count += 1
			if count%100 == 0:
				"""
				pred = predict(s)
				print "Epoch:",epoch
				print ' '.join(id2vocab[idx] if idx != -1 else 'UNK' for idx in s)
				print ' '.join(id2vocab[idx] if idx != len(id2vocab) else 'UNK' for idx in pred)
				pickle.dump([p.get_value() for p in parameters],open('params','wb'),2)
				"""
				pass
		test_score = run_test(vocab2id,test_file,test)
		print
		print "Test result:",test_score
		print
		if test_score < max_test:
			max_test = test_score
			pickle.dump([p.get_value() for p in parameters],open('params','wb'),2)
		else:
			print "Final:",max_test
			exit()

