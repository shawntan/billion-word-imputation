import theano
import sys
import theano.tensor as T
import numpy         as np
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
import cPickle       as pickle
from numpy_hinton import print_arr
from theano.printing import Print
from vocab import read_file

def create_vocab_vectors(P,vocab2id,size):
	P.V   = U.initial_weights(len(vocab2id) + 1,size)
	P.V_b = U.initial_weights(len(vocab2id) + 1)
	return P.V,P.V_b

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

def rae(P,W_input,W_state,inputs,state_0,states):
	P.b_input_rec = U.initial_weights(W_input.get_value().shape[1])
	P.b_state_rec = U.initial_weights(W_state.get_value().shape[1])

	input_rec = T.dot(states,W_input.T) + P.b_input_rec
	state_rec = T.tanh(T.dot(states,W_state.T) + P.b_state_rec)

	input_rec_cost = T.sum((input_rec - inputs)**2)
	state_rec_cost = T.sum((state_rec[1:] - states[:-1])**2)

	cost = input_rec_cost + state_rec_cost

	return [P.b_input_rec,P.b_state_rec],cost





def create_model(ids,vocab2id,size):
	word_vector_size = size
	rae_state_size   = size
	predictive_hidden_size = rae_state_size

	P = Parameters()
	V,b_predict = create_vocab_vectors(P,vocab2id,word_vector_size)
	W_predict = V.T
#	W_predict = U.create_shared(U.initial_weights(predictive_hidden_size,V.get_value().shape[0]))
	X = V[ids]



	# RAE parameters
	P.W_state = U.initial_weights(rae_state_size,rae_state_size)
	P.W_input = U.initial_weights(rae_state_size,rae_state_size)
	P.b_state = U.initial_weights(rae_state_size)
	P.state_0 = U.initial_weights(rae_state_size)
	
	states = recurrent_combine(P.state_0,X,P.W_input,P.W_state,P.b_state)

	scores = T.dot(states,W_predict) + b_predict
	scores = T.nnet.softmax(scores)

	log_likelihood, cross_ent = word_cost(scores[:-1],ids[1:])
	recon_b, rae_cost = rae(P,P.W_input,P.W_state,X,P.state_0,states)


	parameters = P.values()

	cost = log_likelihood + 1e-5 * sum( T.sum(w**2) for w in parameters )
	obv_cost = cross_ent
	return scores, cost, obv_cost, parameters

def make_accumulate_update(inputs,outputs,parameters,gradients,update_method=updates.adadelta):
	acc = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	count = U.create_shared(np.int32(0))
	acc_update = [ (a,a + g) for a,g in zip(acc,gradients) ] + [ (count,count+1) ]
	acc_gradient = theano.function(
				inputs = inputs,
				outputs = outputs,
				updates = acc_update
			)
	avg_gradient = [ a/count for a in acc ]
	clear_update = [ (a,0.*a) for a,g in zip(acc,parameters) ] + [ (count,0) ]
	train_acc = theano.function(
			inputs=[],
			updates=update_method(parameters,avg_gradient) + clear_update
		)
	return acc_gradient,train_acc


def training_model(vocab2id,size):
	ids = T.ivector('ids')
	scores, cost, obv_cost, parameters = create_model(ids,vocab2id,size)


	gradients = T.grad(cost,wrt=parameters)
	print "Computed gradients"
	acc_gradient,train_acc = make_accumulate_update(
			inputs  = [ids],
			outputs = obv_cost,
			parameters = parameters, gradients=gradients,
			update_method=updates.adadelta
		)
	test = theano.function(
			inputs  = [ids],
			outputs = obv_cost
		)

	predict = theano.function(
			inputs  = [ids],
			outputs = T.argmax(scores,axis=1)
		)

	return predict,acc_gradient,train_acc,test,parameters

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

	predict,acc_gradient,train_acc,test,parameters = training_model(vocab2id,20)
	print "Loading params..."
	try:
		saved_params = pickle.load(open('params','rb'))
		for p,sp in zip(parameters,saved_params): p.set_value(sp)
	except:
		pass

	max_test = np.inf
	prev_params = parameters[0].get_value()
	for epoch in range(10):
		count = 0

		for s in sentences(vocab2id,sentence_file):
			s = np.array(s,dtype=np.int32)
			score = acc_gradient(s)
			print score
			count += 1
			if count%50 == 0:
				train_acc()
				curr_params = parameters[0].get_value()
				#print
				#print np.sum((curr_params-prev_params)**2)
				#print
				prev_params = curr_params
				
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
