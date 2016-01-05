import sys
import math
import json

import numpy as np
import neurolab as nl

from chnlp.word2vec.model import Model
from chnlp.altlex.extractSentences import MarkerScorer

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

#learner = Pipeline([('poly', PolynomialFeatures(degree=2)),
#                  ('linear', LinearRegression(fit_intercept=False))])
#learner = LinearRegression()
#learner = Ridge(alpha=.5)
learner = SGDRegressor(verbose=100)

#read in word vector model
model = Model(sys.argv[1])

j = {}
model._load()
count = 0
for word in model._model.vocab:
    vector = [float(i) for i in model._model[word]]
    j[word] = vector
    count += 1
    if count > float('inf'):
        break
import json
with open(sys.argv[1] + '.json', 'w') as f:
    json.dump(j, f)
exit()

#read in each marker word pair file
wordpairs = MarkerScorer()

#change all tf to be 1 + log(tf)
#(because of Zipf's law)
#create dataset by for each word pair concatenating their feature vectors (order matters)
for marker in wordpairs.markerPairs:
    #if marker != 'because':
    #    continue
    print(marker)

    X = []
    y = []
    ds = SupervisedDataSet(400, 1)
    for word1 in wordpairs.markerPairs[marker]:
        vector1 = model.vector(word1)
        if vector1 is None:
            #print("cant find word 1 {}".format(word1))
            continue
        for word2 in wordpairs.markerPairs[marker][word1]:
            vector2 = model.vector(word2)
            if vector2 is None:
                #print("cant find word 2 {}".format(word2))
                continue            
            features = np.append(vector1, vector2)
            #print(len(vector1), len(vector2), len(features))
            X.append(features)
            y.append(math.log(wordpairs.markerPairs[marker][word1][word2],2))
            #y.append(wordpairs.markerPairs[marker][word1][word2])
            #ds.addSample(features, wordpairs.markerPairs[marker][word1][word2])
            
    if len(X) < 2:
        continue
    if True:
        numTrain = int(len(X)/2.0)
        print(numTrain, len(features))
        trainX,testX = X[:numTrain],X[numTrain:]
        trainY,testY = y[:numTrain],y[numTrain:]

        #net = nl.net.newff([[-1, 1] for i in range(400)],[100, 1])
        #net.train(trainX, np.array(trainY).reshape(len(trainY),1), epochs=100, show=1)
        
        learner.fit(trainX, trainY)
        print(marker, learner.score(testX, testY))
        if len(trainX) < 2000000:
            learner.fit(X, y)
        filename = '{}.{}'.format(sys.argv[1], marker)
        #joblib.dump(learner, filename, compress=9)
        coef = list(learner.coef_)
        intercept = list(learner.intercept_)
        params = learner.get_params()
        j = {'coef': coef,
             'intercept': intercept,
             'params': params}
        with open(filename + '.test', 'w') as f:
            json.dump(j,f)

        #now cross validate using different regression models
        #want best model for each marker pair
        #save final model trained on all data

    elif False:
        #ds.setField('input', trainX)
        #ds.setField('target', trainY)

        if 5000 > len(X):
            proportion= 1.0
        else:
            proportion = 5000.0/len(X)
            TrainDS, TestDS = ds.splitWithProportion(proportion)
            net = buildNetwork(400,
                               100, # number of hidden units
                               1,
                               bias = True,
                               hiddenclass = SigmoidLayer,
                               outclass = LinearLayer
                               )
            trainer = BackpropTrainer(net, TrainDS, verbose = True)
            trainer.trainUntilConvergence(maxEpochs = 100)


