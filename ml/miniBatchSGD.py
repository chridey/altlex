import time

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_fscore_support

class MiniBatchSGD(BaseEstimator):
    def __init__(self, verbose=False, test=None, batch_size=100,
                 *args, **kwargs):
        self.classifier = SGDClassifier(*args, **kwargs)
        self.verbose = verbose
        self.test = test
        self.batch_size = batch_size
        
    def fit(self, X, y):
        indices = np.arange(X.shape[0])
        num_batches = int(1.*X.shape[0]/self.batch_size) + ((X.shape[0] % self.batch_size) != 0)
        classes = np.unique(y)
        
        for i in range(self.classifier.n_iter):
            if self.verbose:
                print('epoch {}'.format(i))

            np.random.shuffle(indices)

            for j in range(num_batches):
                X_batch = X[indices[j*self.batch_size:(j+1)*self.batch_size]]
                y_batch = y[indices[j*self.batch_size:(j+1)*self.batch_size]]

                start = time.time()
                self.classifier.partial_fit(X, y, classes)
                y_pred = self.classifier.predict(X)
                p,r,f,s = precision_recall_fscore_support(y, y_pred)
                cost = f
                
                if self.verbose:
                    print ("epoch: {} batch: {} cost: {} time: {}".format(i, j, cost, time.time()-start))
                if j % 10 == 0 and self.verbose and self.test is not None:
                    for index in range(len(self.test)):
                        y_pred = self.classifier.predict(self.test[index][0])
                        p,r,f,s = precision_recall_fscore_support(self.test[index][1], y_pred)
                        print ("precision: {} recall: {} ".format(p, r))

    def predict(self, X):
        return self.classifier.predict(X)

    def decision_function(self, X):
        return self.classifier.decision_function(X)

    def get_params(self):
        params = self.classifier.get_params()
        params.update({'verbose': self.verbose,
                       'test': self.test,
                       'batch_size': self.batch_size})
        return params
