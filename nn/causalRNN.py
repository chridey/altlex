import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
    
    def __init__(self, nh1, nc, de, hl=1): #nh2, nc, de):
        '''
        nh1 :: dimension of the first hidden layer
        nh2 :: dimension of the second hidden layer
        nc :: number of classes
        de :: dimension of the word embeddings
        '''
        # parameters of the model
        #self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
        #           (ne, de)).astype(theano.config.floatX))

        self.Ww  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de, nh1)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh1, nh1)).astype(theano.config.floatX))
        if hl ==2:
            self.Wl  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                       (nh1, nh1)).astype(theano.config.floatX))
            self.Wr  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                       (nh1, nh1)).astype(theano.config.floatX))
            self.W   = theano.shared(numpy.zeros(nh1, dtype=theano.config.floatX))
        else:
            #self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
            #           (2*nh1, nc)).astype(theano.config.floatX))
            self.W   = theano.shared(numpy.zeros(2*nh1, dtype=theano.config.floatX))

        #self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        #self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh1, dtype=theano.config.floatX))

        # bundle
        if hl==1:
            self.params = [ self.Ww, self.Wh, self.W, self.h0]
            self.names  = ['Ww', 'Wh', 'W', 'h0']
        else:
            self.params = [ self.Ww, self.Wh, self.Wl, self.Wr, self.W, self.h0]
            self.names  = ['Ww', 'Wh', 'Wl', 'Wr', 'W', 'h0']

        #for the first half of the sentence
        x1 = T.matrix()
        #x1 = T.tensor3()
        #for the second half of the sentence
        x2 = T.matrix()
        #x2 = T.tensor3()
        #y = T.iscalar('y') #label
        y = T.ivector()
        
        #first do the recurrence
        def recurrence(e_wt, e_tm1):
            e_t = T.nnet.sigmoid(T.dot(e_wt, self.Ww) + T.dot(e_tm1, self.Wh))# + self.bh)
            #s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return e_t

        e_cl1, _ = theano.scan(fn=recurrence, \
            sequences=x1, outputs_info=[self.h0], \
            n_steps=x1.shape[0])
        e_cl2, _ = theano.scan(fn=recurrence, \
            sequences=x2, outputs_info=[self.h0], \
            n_steps=x2.shape[0])
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')
        #nll = -T.mean(T.log(s[0])[y])
        #nll = -T.mean(T.log(s)[T.arange(y.shape[0]), y])
        #nll = -T.mean(T.log(s)[y])

        #now combine the two clauses
        if hl==1:
            y_pred = 1 / (1 + T.exp(-T.dot(T.concatenate([e_cl1[-1,:], e_cl2[-1,:]]), self.W)))
        else:
            e_s = T.nnet.sigmoid(T.dot(e_cl1[-1,:], self.Wl) + T.dot(e_cl2[-1,:], self.Wr))
            #s = T.nnet.softmax(T.dot(e_s, self.W))
            y_pred = 1 / (1 + T.exp(-T.dot(e_s, self.W)))

        xent = -y * T.log(y_pred) - (1-y) * T.log(1-y_pred)
        cost = xent.mean()
        
        gradients = T.grad( cost, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[x1, x2], outputs=y_pred)

        self.train = theano.function( inputs  = [x1, x2, y, lr],
                                      outputs = cost,
                                      updates = updates )

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

    def load(self, folder):
        for param, name in zip(self.params, self.names):
            param.set_value(numpy.load(os.path.join(folder, name + '.npy')))
        
