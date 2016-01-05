from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from chnlp.ml.gridSearch import GridSearchSVM, GridSearchLogit, GridSearchSGD, GridSearchLabelSpreader,GridSearchPipeline

from chnlp.ml.tfkld import TfkldFactorizer

#ignore for now
from chnlp.ml.labelSpreading import LabelSpreader
from chnlp.ml.transductiveSVM import TransductiveSVM
from chnlp.ml.gmm import GMM
from chnlp.mixture.mixtureClient import MixtureClient

try:
    from chnlp.ml.rnnLearner import RNNLearner
except ImportError:
    RNNLearner = None    
#####
    
import chnlp.altlex.featureExtractor as fe

class Config:
    def __init__(self, params=None):
        self.setParams(params)

    @property
    def defaultParams(self):
        return {'classifier': {'name': 'svm',
                               'settings': {}},
                'features': {'experimental': {},#'altlex_stem', 'curr_stem', 'prev_stem'}, #'curr_length', 'prev_length', 'curr_length_post_altlex'}, #'hedging'}, #'altlex_nouns', 'altlex_verbs'},
                             'fixed': self.featureExtractor.defaultSettings,
                             'groups': {}
                             }}

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
        return {'random_forest': RandomForestClassifier,
                'naive_bayes': BernoulliNB,
                'ada_boost': AdaBoostClassifier,
                'svm': SVC,
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
                'rnn': RNNLearner,
                'tfkld': TfkldFactorizer}

    @property
    def classifierSettings(self):
        return {'random_forest': dict(n_estimators=1000,
                                      #tried 10, 100, and 500
                                      #higher number increases precision
                                      n_jobs=1,
                                      oob_score=False,
                                      bootstrap=True,
                                      #False reduces recall
                                      max_features="sqrt",
                                      #using all reduces precision
                                      #log2 reduces recall w/o p incr
                                      #.5 reduces precision (lt max tho)
                                      min_samples_split=2
                                      #1 and 2 no diff
                                      #3 is worse
                                      ),
                'naive_bayes': {},
                'ada_boost': dict(n_estimators=500),
                'svm': dict(verbose=True,
                            #kernel='poly',
                            #C=10, gamma=1)
                            #C=.01, gamma=.1)
                            #C=.01)
                            C=10),
                            #degree=3),
                'logistic_regression': dict(C=.1),
                'grid_search_svm': {},
                'grid_search_logistic_regression': {},
                'grid_search_sgd': {},
                'grid_search_label_spreading': {},
                'grid_search_pipeline': {},
                'tfkld': {},
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

    @property
    def groupSettings(self):
        return self.params['features']['groups']

    def featureSubset(self, *featureSubsets):
        ret = {}
        for featureSubset in featureSubsets:
            ret.update(self.featureExtractor.featureSubsets(featureSubset))
        return ret
