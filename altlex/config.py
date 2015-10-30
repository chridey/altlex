from chnlp.ml.randomForest import RandomForest
from chnlp.ml.naiveBayes import NaiveBayes
from chnlp.ml.adaBoost import AdaBoost
from chnlp.ml.svm import SVM
from chnlp.ml.logisticRegression import LogisticRegression

from chnlp.ml.gridSearch import GridSearchSVM, GridSearchLogit, GridSearchSGD, GridSearchLabelSpreader,GridSearchPipeline

from chnlp.ml.labelSpreading import LabelSpreader
from chnlp.ml.transductiveSVM import TransductiveSVM

try:
    from chnlp.ml.rnnLearner import RNNLearner
except ImportError:
    RNNLearner = None
    
from chnlp.ml.gmm import GMM

from chnlp.mixture.mixtureClient import MixtureClient

import chnlp.altlex.featureExtractor as fe
#from chnlp.altlex.featureExtractor import FeatureExtractor

class Config:
    def __init__(self, params=None):
        self.setParams(params)

    @property
    def defaultParams(self):
        return {'classifier': {'name': 'svm',
                               'settings': {}},
                'features': {'experimental': {},#'altlex_stem', 'curr_stem', 'prev_stem'}, #'curr_length', 'prev_length', 'curr_length_post_altlex'}, #'hedging'}, #'altlex_nouns', 'altlex_verbs'},
                             'fixed': self.featureExtractor.defaultSettings}}

    def setParams(self, params):
        if params is None:
            self.featureExtractor = fe.FeatureExtractor()
            self.params = self.defaultParams
        else:
            self.validate(params)
            self.params = params

            if 'cacheSize' in params:
                fe.cacheSize = params['cacheSize']
            
            self.featureExtractor = fe.FeatureExtractor()

    def setFeatures(self, features):
        features = set(features)
        for feature in features:
            assert(feature in self.featureExtractor.validFeatures)
        for f in self.params['features']['fixed']:
            if f in features:
                self.params['features']['fixed'][f] = True
            else:
                self.params['features']['fixed'][f] = False
                
    @property
    def classifiers(self):
        return {'random_forest': RandomForest,
                'naive_bayes': NaiveBayes,
                'ada_boost': AdaBoost,
                'svm': SVM,
                'logistic_regression': LogisticRegression,
                'grid_search_svm': GridSearchSVM,
                'grid_search_logistic_regression': GridSearchLogit,
                'grid_search_sgd': GridSearchSGD,
                'grid_search_label_spreading': GridSearchLabelSpreader,
                'grid_search_pipeline': GridSearchPipeline,
                'label_spreading': LabelSpreader,
                'gmm': GMM,
                'transductive_svm': TransductiveSVM,
                'constrained_gmm': MixtureClient,
                'rnn': RNNLearner}

    @property
    def semisupervised(self):
        return {'label_spreading',
                'grid_search_label_spreading',
                'transductive_svm',
                'constrained_gmm'}

    @property
    def unsupervised(self):
        return {'gmm'}

    def validate(self, params):
        assert(type(params) == dict)
        assert('classifier' in params)
        assert('name' in params['classifier'])
        assert('settings' in params['classifier'])
        assert(params['classifier']['name'] in self.classifiers)

        assert('features' in params)
        assert('experimental' in params['features'])
        assert('fixed' in params['features'])

        for feature in params['features']['experimental']:
            assert(feature in self.featureExtractor.validFeatures)

        for feature in params['features']['fixed']:
            assert(feature in self.featureExtractor.validFeatures)

    @property
    def classifier(self):
        return self.params['classifier']['name']

    @property
    def experimentalSettings(self):
        return self.params['features']['experimental']

    @property
    def fixedSettings(self):
        return self.params['features']['fixed']

    def featureSubset(self, *featureSubsets):
        ret = {}
        for featureSubset in featureSubsets:
            ret.update(self.featureExtractor.featureSubsets(featureSubset))
        return ret
