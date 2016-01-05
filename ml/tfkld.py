from __future__ import print_function

import sys
import math

import numpy

import scipy.sparse as ssp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import NMF, TruncatedSVD

def identity(x):
    return x

class TfkldTransformer:
    def __init__(self, input='content', ngram_range=(1,1), verbose=False):
        self.count_vectorizer = CountVectorizer(input=input, ngram_range=ngram_range, dtype=numpy.float, tokenizer=identity, preprocessor=identity)
        self.weight = None
        self.verbose = verbose
        
    def fit(self, X, y):
        assert(len(X) // 2 == len(y))
        if self.verbose:
            print('Creating word counts')
        X = self.count_vectorizer.fit_transform(X).todense()
        n_docs, n_words = X.shape
        if self.verbose:
            print(X.shape)
        # (0, F), (0, T), (1, F), (1, T)
        count = numpy.ones((4, n_words))
        if self.verbose:
            print('Creating weight counts')                
        for n in range(0, n_docs, 2):
            if n % 1000  == 0:
                if self.verbose:
                    print ('Processed {} rows'.format(n))
            for d in range(n_words):
                label = y[n // 2]
                if ((X[n,d] > 0) and (X[n+1,d] == 0)) or ((X[n,d] == 0) and (X[n+1,d] > 0)):
                    # Non-shared
                    if label == 0:
                        # (0, F)
                        count[0,d] += 1.0
                    elif label == 1:
                        # (1, F)
                        count[2,d] += 1.0
                elif (X[n,d] > 0) and (X[n+1,d] > 0):
                    # Shared
                    if label == 0:
                        # (0, T)
                        count[1,d] += 1.0
                    elif label == 1:
                        # (1, T)
                        count[3,d] += 1.0

        if self.verbose:
            print('Computing KLD')
        self.weight = ssp.lil_matrix(self.computeKLD(count))

        return self
    
    def transform(self, X):
        assert(self.weight is not None)
        X = self.count_vectorizer.transform(X)

        for n in range(X.shape[0]):
            X[n, :] = X[n, :].multiply(self.weight)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def computeKLD(self, count):
        # Smoothing
        count = count + 0.05
        # Normalize
        pattern = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
        pattern = numpy.array(pattern)
        prob = count / (pattern.dot(count))
        #
        ratio = numpy.log((prob[0:2,:] / prob[2:4,:]) + 1e-7)
        return (ratio * prob[0:2,:]).sum(axis=0)

    def save(self, filename):
        joblib.dump(self, filename)

class KldTransformer:
    def __init__(self):
        self.dict_vectorizer = None
        
    def fit(self, p_num, p_denom, q_num, q_denom, smoothing=1):
        self.kld = {}
        self._p = {}
        self._q = {}
        for feat in p_num:
            p = 1. * (smoothing + p_num[feat]) / (smoothing + p_denom[feat])
            q = 1. * (smoothing + q_num[feat]) / (smoothing + q_denom[feat])
            
            if p <= 0 or p >= 1 or q <= 0 or q >= 1:
                kl_pq = 10
            else:
                kl_pq = p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))
            if kl_pq > 10:
                kl_pq = 10
                
            self.kld[feat] = kl_pq
            self._p[feat] = p
            self._q[feat] = q

    def topKLD(self, num=None, greater=True, lesser=True, prefix=''):
        if num is None:
            num = len(self.kld)
        ret = []
        for i,(feat,score) in enumerate(sorted(self.kld.iteritems(), key=lambda x:x[1], reverse=True)):
            if i > num:
                break
            if type(feat) == tuple and not feat[0].startswith(prefix):
                continue
            if self._p[feat] <= self._q[feat]:
                if lesser:
                    ret.append((feat, self._p[feat], self._q[feat], self.kld[feat]))
            if self._p[feat] >= self._q[feat]:
                if greater:
                    ret.append((feat, self._p[feat], self._q[feat], self.kld[feat]))
        return ret
    
    def save(self, filename):
        joblib.dump(self, filename)
        
    def transform(self, X):
        #multiply the feature TFs by their KLD weights
        print('in kld')
        print(len(self.kld))
        X_new = []
        for x in X:
            x_new = {}
            for key in x.keys():
                #print(key)
                if key in self.kld:
                    x_new[key] = x[key]*self.kld[key]
                    #print(self.kld[key], x[key], x_new[key])
            X_new.append(x_new)

        #use a dict vectorizer to convert them to sparse arrays
        if self.dict_vectorizer is None:
            self.dict_vectorizer = DictVectorizer()
            return self.dict_vectorizer.fit_transform(X_new)
        return self.dict_vectorizer.transform(X_new)

class TfkldFactorizer:
    def __init__(self,
                 weight_filename,
                 max_iter=20,
                 n_components=100,
                 factorizer='svd'):
        self.transformer = joblib.load(weight_filename)
        if factorizer == 'svd':
            self.factorizer = TruncatedSVD(n_components=n_components,
                                           n_iter=max_iter)
        else:
            self.factorizer = NMF(n_components=n_components, max_iter=max_iter)
            
    def fit(self, X, *y):
        self.factorizer.fit(X)
        return self
    
    def transform(self, X):
        X = self.transformer.transform(X)
        print(X.shape)
        return self.factorizer.transform(X)

    def fit_transform(self, X, *y):
        X = self.transformer.transform(X)
        return self.factorizer.fit_transform(X)

    def save(self, filename, *args):
        joblib.dump(self.factorizer, filename + '.factorizer')
        numpy.savez(filename, *args)

    def load(self, filename):
        self.factorizer = joblib.load(filename + '.factorizer')
        return numpy.load(filename)        

