from sklearn.externals import joblib

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.semi_supervised import LabelSpreading,LabelPropagation

from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import StratifiedKFold

from sklearn import metrics

import numpy as np

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.base import BaseEstimator

from chnlp.ml.sklearner import Sklearner

class GridSearch(BaseEstimator):
    def __init__(self, classifier, parameters, searchParameters):
        #avoid circular dependencies
        from chnlp.altlex.config import Config
        self.config = Config()
        self.classifierType = self.config.classifiers[classifier]
        self.parameters = parameters
        self.searchParameters = searchParameters
        self.setClassifier()
        
    def setClassifier(self):
        #skf = StratifiedKFold(y,
        #                      n_folds=2)

        self.classifier = GridSearchCV(self.classifierType(**self.parameters),
                                       self.searchParameters,
                                       scoring = 'f1',
                                       cv=4,
                                       n_jobs=8,
                                       verbose=1)
                                       #cv=skf)

    '''
    def train(self, features, transform=True):
        #super().train(features, transform)
        Sklearner.train(self,features, transform)
        print(self.classifier.best_score_)
        print(self.classifier.best_params_)
        print(self.classifier.grid_scores_)
        self.model = self.classifier.best_estimator_
        return self.model
    '''
    def fit(self, X, y):
        model = self.classifier.fit(X, y)
        print(self.classifier.best_score_)
        print(self.classifier.best_params_)
        print(self.classifier.grid_scores_)
        return model

def _evaluateParameters(X, y, X_tune, y_tune,
                        classifierType, fixedParameters, searchParameters,
                        scoring):
        parameters = fixedParameters.copy()
        parameters.update(searchParameters)
        classifier = classifierType(**parameters)
        classifier.fit(X, y)

        y_tune_predict = classifier.predict(X_tune)

        if scoring == 'f1':
            return classifier, metrics.f1_score(y_tune, y_tune_predict)
        else:
            raise NotImplementedError

class HeldOutGridSearch(Sklearner, GridSearch):
    def __init__(self, classifier, parameters, searchParameters,
                 transformer=None, preprocessor=None, tuning=None,
                 n_jobs=8, scoring = 'f1', verbose=False):
        GridSearch.__init__(self, classifier, parameters, searchParameters)
        Sklearner.__init__(self, self.classifier, transformer, preprocessor)
        self.setTuning(tuning)
        self.parallel = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)
        self.scoring = scoring
        self.verbose = verbose

    def setClassifier(self):
        self.classifier = self.classifierType(**self.parameters)

    def setTuning(self, tuning):
        if tuning is not None:
            self.X_tune, self.y_tune = zip(*tuning)
        else:
            self.X_tune, self.y_tune = None, None
        self.X_tune_transformed = False
        
    def transform(self, features, *args, **kwargs):
        X = Sklearner.transform(self, features, *args, **kwargs)
        if not self.X_tune_transformed:
            self.X_tune = Sklearner.transform(self, self.X_tune)
            self.X_tune_transformed = True
        return X
    
    def fit(self, X, y):
        assert(self.X_tune is not None)
        if self.verbose:
            print('grid searching...')
        #ret = list((self._evaluateParameters(X, y, params) for params in ParameterGrid(self.searchParameters)))
        ret = self.parallel(joblib.delayed(_evaluateParameters)(X, y,
                                                               self.X_tune, self.y_tune,
                                                               self.classifierType,
                                                               self.parameters,
                                                               params,
                                                               self.scoring) for params in ParameterGrid(self.searchParameters))
        
        if len(ret):
            bestClassifier, bestScore = max(ret, key=lambda x:x[1])
            if self.verbose:
                for c,s in ret:
                    print({i:c.get_params()[i] for i in set(c.get_params()) & set(self.searchParameters)},
                          s)
            print("Best: {} Score: {}".format({i:bestClassifier.get_params()[i] for i in set(bestClassifier.get_params()) & set(self.searchParameters)}, bestScore))

            self.classifier = bestClassifier
            
                 
