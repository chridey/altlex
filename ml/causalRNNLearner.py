from __future__ import print_function

import random

import gensim

import numpy

from sklearn.base import BaseEstimator

from chnlp.nn.causalRNN import model as causalModel

class RNNClassifier(BaseEstimator):
    def __init__(self,
                 hiddenStates=100,
                 hiddenLayers=2,
                 seed=31415,
                 learningRate=.05,
                 epochs=50,
                 verbose=True
                 ):

        self.hiddenStates = hiddenStates
        self.hiddenLayers = hiddenLayers
        self.learningRate = learningRate
        self.epochs = epochs
        self.verbose = verbose
        self.model = gensim.models.Word2Vec.load('/local/nlp/chidey/model.1428650472.31.word2vec.curr.nelemmas')
        numpy.random.seed(settings['seed'])
        random.seed(settings['seed'])
        self.causalModel = causalModel(self.hiddenStates,
                                       2,
                                       self.model.layer1_size,
                                       self.hiddenLayers)
        
    def get_params(self, deep=True):
        return {'hiddenStates': self.hiddenStates,
                'hiddenLayers': self.hiddenLayers,
                'learningRate': self.learningRate,
                'epochs': self.epochs
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
        
        return self

    def score(self, data, class_values):
        
        return score
        
    def predict(self, data):
        
        return labels
    
class RNNLearner(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = TransductiveSVMClassifier(C=10,
                                                    positive_fraction=.2)

    def metrics(self, testing, transform=True):
        truepos = 0
        trueneg = 0
        falsepos = 0
        falseneg = 0

        features, labels = zip(*testing)
        if transform:
            X = self._transform(features)
        else:
            X = features

        assigned = self.classifier.predict(X)
              
        for i,label in enumerate(labels):
            if assigned[i] == label:
                if label == True:
                    truepos += 1
                else:
                    trueneg += 1
            elif label == False:
                falsepos +=1
            else:
                falseneg += 1

        print(truepos, trueneg, falsepos, falseneg)

        precision, recall = self._calcPrecisionAndRecall(truepos, trueneg, falsepos, falseneg)
        
        return precision, recall
            
        
    
    
