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

from chnlp.ml.sgd import SGD
from chnlp.ml.labelSpreading import LabelSpreader,LabelPropagator
from chnlp.ml.sklearner import Sklearner

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.base import BaseEstimator

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
            
class GridSearchSVM(GridSearch):
    def __init__(self):
        #super().__init__()
        GridSearch.__init__(self)
        self.classifierType = SVC#(kernel='linear')#(verbose=True)
        self.parameters = {}#'kernel': 'linear'}
        self.searchParameters = {'C': (.1, 1, 10, 100, 1000),
                           }#'gamma': (0, .0001, .001, .01, .1)} #, 1)}
        self.setClassifier()

class GridSearchLinearSVM(GridSearch):
    def __init__(self):
        #super().__init__()
        GridSearch.__init__(self)
        self.classifierType = LinearSVC#(kernel='linear')#(verbose=True)
        self.parameters = {}
        self.searchParameters = {'C': (.1, 1, 10, 100, 1000),
                           }#'gamma': (0, .0001, .001, .01, .1)} #, 1)}
        self.setClassifier()

class GridSearchLogit(GridSearch):
    def __init__(self):
        #super().__init__()
        GridSearch.__init__(self)
        self.classifierType = LogisticRegression
        self.parameters = {}
        self.searchParameters = {'C': (.01, .1, 1, 10, 100)}
        self.setClassifier()

class GridSearchSGD(GridSearch):
    def __init__(self):
        #super().__init__()
        GridSearch.__init__(self,
                            'sgd',
                            {'n_iter': 100},
                            {'penalty': ('l2', 'elasticnet'),
                           'alpha': 10.0**-np.arange(1,7)} #(0.001, 0.0001, 0.00001)}
                            )
        
class GridSearchLabelSpreader(GridSearch, LabelSpreader):
    def __init__(self):
        super().__init__()
        self.classifierType = LabelSpreading
        self.parameters = {}
        self.searchParameters = {'gamma': (.02, .2, 2, 20, 200),
                           'alpha': (.05, .1, .2, .5, .8, 1)}
        self.setClassifier()
        
    def show_most_informative_features(self, n=50):
        #TODO
        pass
    
class GridSearchPipeline(GridSearch):
    choices = {'svm': GridSearchSVM,
               'sgd': GridSearchSGD,
               'lr': GridSearchLogit}
    def __init__(self, choice='sgd'):
        assert(choice in self.choices)
        clf = self.choices[choice]()
        self.vectorizer = clf.vectorizer
        self.classifierType = Pipeline
        self.parameters = {'steps': [('chi', SelectKBest(chi2)),
                                     (choice, clf.classifierType(**clf.parameters))]}

        self.searchParameters = {'chi__k': (1000, 'all')}
        self.searchParameters.update({'{}__{}'.format(choice, param):clf.searchParameters[param] for param in clf.searchParameters})
        self.setClassifier()

    @property
    def _feature_importances(self):
        return self.classifier.best_estimator_.steps[1][1].coef_.tolist()[0]
'''
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
#    'chi__k': (10000, 100000, 'all'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }
'''
                 
