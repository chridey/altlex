from __future__ import print_function

import os
import json
import collections
import pickle

import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support

from sklearn.utils import shuffle

from scipy.stats import ttest_rel

from chnlp.utils.utils import indexedSubset, balance

class Sklearner:
    def __init__(self, classifier, transformer=None, preprocessor=None):
        #self.featureMap = None
        #self.reverseFeatureMap = None
        self.classifier = classifier
        self.transformer = transformer
        if self.transformer is None:
            self.transformer = getattr(self.classifier, 'transformer', None)
        if self.transformer is None:
            self._forced = True
        else:
            self._forced = False
        if preprocessor is None:
            class IdentityPreprocessor:
                def fit_transform(self, X): return X
            self.preprocessor = IdentityPreprocessor()
        else:
            self.preprocessor = preprocessor

        print(type(self.transformer))
        
    def transform(self, features, force=False):
        if self.transformer is None or (force and self._forced):
            self.transformer = DictVectorizer()
            return self.preprocessor.fit_transform(self.transformer.fit_transform(features).toarray())
            
        return self.preprocessor.fit_transform(self.transformer.transform(features).toarray())
    
    def fit(self, X, y):
        #shuffle the data, important for classifiers such as SGD
        X, y = shuffle(X, y, random_state=0)

        y = np.array(y, dtype=int)
        self.model = self.classifier.fit(X, y)
        return self.model

    def fit_transform(self, X, y):
        X = self.transform(X, True)
        print(X.shape)
        return self.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def metrics(self, X, y):
        X = self.transform(X)
        y = np.array(y, dtype=int)
        y_predict = self.predict(X)
        accuracy = 1. * sum(y_predict == y) / y.shape[0]
        precision, recall, f_score, support = precision_recall_fscore_support(y, y_predict)
        return accuracy, precision, recall, f_score,  y_predict

    def crossvalidate(self,
                      X,
                      y,
                      n_folds=2,
                      training=(None,None),
                      balanced=True,
                      printResults=False,
                      printErrorAnalysis=False):

        tX, ty = training
        y = np.array(y)
        if tX is not None:
            X_trans = self.transform(X + tX)
            tX_trans = self.transform(tX)
        else:
            X_trans = self.transform(X)

        skf = StratifiedKFold(y,
                              n_folds=n_folds,
                              random_state=1) #make sure we always use same data
    
        accuracy = []
        precisions = []
        recalls = []
        print(X_trans.shape)
        for index, (train_index,test_index) in enumerate(skf):
            X_train = X_trans[train_index]
            y_train = y[train_index]
            print(sum(y_train))
            #need to oversample here
            balancedData = list(zip(X_train, y_train))
            if balanced:
                balancedData = balance(balancedData)

            #now combine with any data that should always be in training
            X_train, y_train = list(zip(*balancedData))
            if tX is not None:
                X_train += tX_trans
                y_train += ty

            print(len(X_train), len(y_train))

            clf = self.fit(np.array(X_train), y_train)
            
            X_test = X_trans[test_index]
            y_test = y[test_index]

            accuracy_, precision, recall, f_score = self.metrics(X_test, y_test)
            accuracy.append(accuracy_)
            precisions.append(precision[1])
            recalls.append(recall[1])

            if printErrorAnalysis:
                self.printErrorAnalysis(indexedSubset(X, set(test_index)),
                                        y_test,
                                        y_predict,
                                        suffix='_{}'.format(index))

        if printResults:
            self.printResults(accuracy, precisions, recalls)
        
        #train the final classifier on all the data
        #need to sample again
        X_final,y_final = X[:len(y)],y
        if balanced:
            X_final,y_final = zip(*balance(list(zip(X_final, y_final))))
        if tX is not None:
            X_final += tX
            y_final += ty
        return self.fit_transform(np.array(X_final), y_final)

    def printErrorAnalysis(self,
                           X,
                           y_true,
                           y_predict,
                           suffix='',
                           dir = ''
                           ):

        false_positives = []
        false_negatives = []
        true_positives = []
        
        for i, dataPoint in enumerate(X):
            if y_true[i] != y_predict[i]:
                if y_predict[i] == False:
                    false_positives.append(dataPoint.data)
                else:
                    false_negatives.append(dataPoint.data)                    
            elif label==True:
                true_positives.append(dataPoint.data)
                
        with open(os.path.join(dir, 'false_positives' + suffix), 'w') as fp:
            json.dump(false_positives, fp)
            
        with open(os.path.join(dir, 'false_negatives' + suffix), 'w') as fn:
            json.dump(false_negatives, fn)
                
        with open(os.path.join(dir, 'true_positives' + suffix), 'w') as fn:
            json.dump(true_positives, fn)
                

    def printResults(self, accuracy, precision, recall): #handle=sys.stdout
        if type(accuracy) == list:
            n_folds = len(accuracy)
            f_measures = [2.0 * precision[i] * recall[i] / (precision[i]+recall[i]) for i in range(n_folds)]
            print(f_measures)
            accuracy = sum(accuracy)
            precision = sum(precision)
            recall = sum(recall)
        else:
            n_folds = 1

        n_folds*=1.0
        accuracy /= n_folds
        precision /= n_folds
        recall /= n_folds

        f_measure = 2 * precision * recall / (precision+recall)
        print('''
              Accuracy: {}
              True precision: {}
              True recall: {}
              True F-measure: {}
              '''.format(
                  accuracy,
                  precision,
                  recall,
                  f_measure
                  ))

        return f_measure
            
    def prob(self, features, transform=True):
        raise NotImplementedError

    def confidence(self, features, transform=True):
        raise NotImplementedError
    
    @property
    def numClasses(self):
        return self.model.classes_
    
    @property
    def _feature_importances(self):
        return self.model.feature_importances_

    def show_most_informative_features(self, n=50):
        '''
        if self.featureMap is None:
            return None
        if self.reverseFeatureMap is None:
            self.reverseFeatureMap = dict((v,k) for k,v in self.featureMap.items())
            self.featureImportances = []
            for i,f in enumerate(self._feature_importances):
                self.featureImportances.append((self.reverseFeatureMap[i], f))

        '''

        if self.transformer is None:
            return None

        self.featureImportances = []
        print(len(self._feature_importances))
        for i,f in enumerate(self._feature_importances):
            self.featureImportances.append((self.transformer.get_feature_names()[i],
                                           f))
        self.featureImportances = sorted(self.featureImportances,
                                         key=lambda x:abs(x[1]),
                                         reverse=True)
            
        for featureName,featureValue in self.featureImportances[:n]:
            print("{}\t{}".format(featureName, featureValue))
        
         #print(self.model.feature_importances_)

    def save(self, filename):
        try:
            joblib.dump(self.model, filename, compress=9)
            joblib.dump(self.transformer, filename + '_vectorizer', compress=9)
        except Exception as e:
            print("ERROR: {}".format(e))
            joblib.dump(self.model, filename)
            joblib.dump(self.transformer, filename + '_vectorizer')
        #with open(filename + '.map', 'wb') as f:
        #    pickle.dump(self.featureMap, f)

    @classmethod
    def load(cls, filename):
        model = joblib.load(filename)
        classifier = cls(model)        
        classifier.transformer = joblib.load(filename + '_vectorizer')
        classifier.model = classifier.classifier
        return classifier

    def close(self):
        #do any cleanup
        self.featureMap = None
        self.reverseFeatureMap = None



    
