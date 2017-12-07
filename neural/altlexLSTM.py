import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.objectives import binary_crossentropy, aggregate
from lasagne.layers import *
from customLayers import *

GRAD_CLIP = 100
DROP_OUT_RATE = 0.5

class AltlexLSTM(object):

    def __init__(self, M, D, embed_caps, d, We):
        
        # theano.config.exception_verbosity='high'

        # input variables
        lem_mat = T.imatrix('lemma_input')
        pos_mat = T.imatrix('pos_input')
        dep_mat = T.imatrix('depenency_input')
        frm_mat = T.imatrix('frame_input')
        alt_mat = T.imatrix('alt_dir_input')
        dir_mat = T.imatrix('dep_dir_input')
        gold = T.ivector('gold')
        m_mat = T.imatrix('mask')
        W = T.ivector('W')
        start_idxs = T.ivector('start_idxs')
        end_idxs = T.ivector('end_idxs')        

        # input layers BxM
        lem_in = InputLayer(shape=(None, M), input_var=lem_mat)
        pos_in = InputLayer(shape=(None, M), input_var=pos_mat)
        dep_in = InputLayer(shape=(None, M), input_var=dep_mat)
        frm_in = InputLayer(shape=(None, M), input_var=frm_mat)
        alt_in = InputLayer(shape=(None, M), input_var=alt_mat)
        dir_in = InputLayer(shape=(None, M), input_var=dir_mat)
        m_in = InputLayer(shape=(None, M), input_var=m_mat)

        # embeddings layers
        lem_embed = EmbeddingLayer(lem_in, input_size=D[0], output_size=embed_caps[0], W=lasagne.utils.floatX(We))
        pos_embed = EmbeddingLayer(pos_in, input_size=D[1], output_size=embed_caps[1])
        dep_embed = EmbeddingLayer(dep_in, input_size=D[2], output_size=embed_caps[2])
        frm_embed = EmbeddingLayer(frm_in, input_size=D[3], output_size=embed_caps[3])
        alt_embed = EmbeddingLayer(alt_in, input_size=D[4], output_size=embed_caps[4])
        dir_embed = EmbeddingLayer(dir_in, input_size=D[5], output_size=embed_caps[5])

        # concat embeddings
        concat_embed = ConcatLayer([lem_embed, pos_embed, dep_embed, frm_embed, alt_embed, dir_embed], axis=2)

        # forward lstm layer
        lstm_forward = LSTMLayer(concat_embed,
                                 num_units=d,
                                 nonlinearity=lasagne.nonlinearities.tanh, 
                                 grad_clipping=GRAD_CLIP, 
                                 mask_input=m_in)
        
        # backward lstm layer
        lstm_backward = LSTMLayer(concat_embed,
                                  num_units=d,
                                  nonlinearity=lasagne.nonlinearities.tanh, 
                                  grad_clipping=GRAD_CLIP, 
                                  backwards=True,
                                  mask_input=m_in)

        lstm_forward_dropout = DropoutLayer(lstm_forward, DROP_OUT_RATE)
        lstm_backward_dropout = DropoutLayer(lstm_backward, DROP_OUT_RATE)

        # slice layer
        forward_last = SliceLayer(lstm_forward_dropout, indices=-1, axis=1)
        backward_last = SliceLayer(lstm_backward_dropout, indices=0, axis=1)
        
        # slice altlex layer
        forward_altlex = AltlexSliceLayer(lstm_forward_dropout, end_idxs)
        backward_altlex = AltlexSliceLayer(lstm_backward_dropout, start_idxs)

        # average output layer
        # forward_avg = AverageOutputLayer(lstm_forward_dropout, m_mat)
        # backward_avg = AverageOutputLayer(lstm_backward_dropout, m_mat)

        # concat lstm outputs
        concat_lstm = ConcatLayer([forward_altlex, backward_altlex, forward_last, backward_last], axis=1)

        # dense layer 1
        dense_out = DenseLayer(concat_lstm, 
                               num_units=d, 
                               nonlinearity=lasagne.nonlinearities.rectify)
        
        # dense layer 2
        self.network = DenseLayer(dense_out, 
                                  num_units=1,
                                  nonlinearity=T.nnet.sigmoid)

        # generate prediction using network
        predictions = get_output(self.network).ravel()

        # calculate loss
        loss = binary_crossentropy(predictions, gold)

        params = get_all_params(self.network, trainable=True)

        # aggregate weighted loss
        loss = aggregate(loss, W, 'normalized_sum')

        updates = lasagne.updates.adam(loss, params, learning_rate=0.001)
        # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        print "Compiling train..."
        self.train = theano.function([lem_mat,
                                      pos_mat, 
                                      dep_mat, 
                                      frm_mat, 
                                      alt_mat, 
                                      dir_mat, 
                                      gold, 
                                      m_mat, 
                                      W, 
                                      start_idxs,
                                      end_idxs],
                                     loss, 
                                     updates=updates,
                                     allow_input_downcast=True)
        
        print "Compiling predict..."
        test_predictions = get_output(self.network, deterministic=True).ravel()
        self.predict = theano.function([lem_mat, 
                                        pos_mat, 
                                        dep_mat, 
                                        frm_mat, 
                                        alt_mat, 
                                        dir_mat, 
                                        m_mat, 
                                        start_idxs,
                                        end_idxs],
                                       test_predictions, 
                                       allow_input_downcast=True)

    def save_params(self, filename):
        param_values = get_all_param_values(self.network)
        np.savez_compressed(filename, *param_values)
