from __future__ import print_function

import random

import gensim

import numpy

from sklearn.base import BaseEstimator

from chnlp.nn.causalRNN import model as causalModel
from chnlp.ml.sklearner import Sklearner

class RNNClassifier(BaseEstimator):
    def __init__(self,
                 hiddenStates=100,
                 embeddingDimension=100,
                 hiddenLayers=2,
                 seed=31415,
                 learningRate=.05,
                 epochs=25,
                 verbose=True
                 ):

        self.hiddenStates = hiddenStates
        self.embeddingDimension = embeddingDimension
        self.hiddenLayers = hiddenLayers
        self.learningRate = learningRate
        self.epochs = epochs
        self.verbose = verbose
        self.seed = seed
        
        numpy.random.seed(seed)
        random.seed(seed)
        self.causalModel = causalModel(self.hiddenStates,
                                       2,
                                       self.embeddingDimension,
                                       self.hiddenLayers)
        
    def get_params(self, deep=True):
        return {'hiddenStates': self.hiddenStates,
                'hiddenLayers': self.hiddenLayers,
                'learningRate': self.learningRate,
                'epochs': self.epochs,
                'seed': self.seed
                }

    def set_params(self, **params):
        if 'hiddenStates' in params:
            self.hiddenStates = params['hiddenStates']
        if 'hiddenLayers' in params:
            self.hiddenLayers = params['hiddenLayers']
        if 'learningRate' in params:
            self.learningRate = params['learningRate']
        if 'epochs' in params:
            self.epochs = params['epochs']
        return self

    def fit(self, data, class_values):
        for i in range(self.epochs):
            if self.verbose:
                print('iteration {}'.format(i))
            for d in range(len(data)):
                if d % 1000 == 0:
                    print('point {}'.format(d))
                x1, x2 = data[d]
                y = class_values[d]
                if not len(x1) or not len(x2):
                    continue
                self.causalModel.train(x1, x2, [y], self.learningRate)
            if self.verbose:
                print("scoring...")
                print(self.score(data, class_values))
        return self

    def _binarize(self, prediction):
        if prediction>.5:
            return 1
        else:
            return 0

    def score(self, data, class_values):
        truePositives = 0
        falsePositives = 0
        trueNegatives = 0
        falseNegatives = 0
        totalCorrect = 0
        for d in range(len(data)):
            x1, x2 = data[d]
            if not len(x1) or not len(x2):
                continue
            label = class_values[d]
            prediction = self._binarize(self.causalModel.classify(x1, x2))
            #print(label, prediction)
            if label == prediction:
                totalCorrect += 1

        return 1.0*totalCorrect/len(data)
        
    def predict(self, data):
        labels = []
        for d in range(len(data)):
            x1, x2 = data[d]
            if len(x1) and len(x2):
                labels.append(self._binarize(self.causalModel.classify(x1, x2)))
            else:
                labels.append(-1)
        return labels
    
class RNNLearner(Sklearner):
    def __init__(self):
        #super().__init__()
        Sklearner.__init__(self)
        self.word2vec = gensim.models.Word2Vec.load('/local/nlp/chidey/model.1428650472.31.word2vec.curr.nelemmas')
        self.classifier = RNNClassifier(embeddingDimension=self.word2vec.layer1_size)
        self.punct = {'!', '-', ',', '.', '?'}

    def _transform(self, features):
        X = []
        for feature in features:
            x1 = [self.word2vec[w.lower()] for w in feature['prev_lemmas'] if w.lower() in self.word2vec and w not in self.punct]
            x2 = [self.word2vec[w.lower()] for w in feature['curr_lemmas'] if w.lower() in self.word2vec and w not in self.punct]
            #if len(x1) and len(x2):
            X.append((x1,x2))
            if not len(x1) or not len(x2):
                print("Bad input: {} {}".format(' '.join(feature['prev_lemmas']),
                                                ' '.join(feature['curr_lemmas'])))
        return X
