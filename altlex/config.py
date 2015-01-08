from chnlp.ml.randomForest import RandomForest
from chnlp.ml.naiveBayes import NaiveBayes
from chnlp.ml.adaBoost import AdaBoost
from chnlp.ml.svm import SVM
from chnlp.ml.logisticRegression import LogisticRegression

from chnlp.ml.gridSearch import GridSearchSVM, GridSearchLogit

from chnlp.altlex.featureExtractor import FeatureExtractor

class Config:
    def __init__(self, params=None):
        self.featureExtractor = FeatureExtractor()

        if params is None:
            self.params = self.defaultParams
        else:
            self.setParams(params)

    @property
    def defaultParams(self):
        return {'classifier': {'name': 'svm',
                               'settings': {}},
                'features': {'experimental': {},
                             'fixed': self.featureExtractor.defaultSettings}}

    def setParams(self, params):
        self.validate(params)
        self.params = params
    
    @property
    def classifiers(self):
        return {'random_forest': RandomForest,
                'naive_bayes': NaiveBayes,
                'ada_boost': AdaBoost,
                'svm': SVM,
                'logistic_regression': LogisticRegression,
                'grid_search_svm': GridSearchSVM,
                'grid_search_logistic_regression': GridSearchLogit}

    def validate(self, params):
        assert(type(params) == dict)
        assert('classifier' in params)
        assert('name' in params['classifier'])
        assert('settings' in params['classifier']['name'])
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
