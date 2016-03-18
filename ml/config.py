from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
try:
    from sklearn.neural_network import MLPClassifier
except ImportError:
    MLPClassifier = None
    
from altlex.ml.miniBatchSGD import MiniBatchSGD

class Config:
            
    @property
    def classifiers(self):
        return {'random_forest': RandomForestClassifier,
                'naive_bayes': BernoulliNB,
                'ada_boost': AdaBoostClassifier,
                'svm': SVC,
                'linear_svm': LinearSVC,
                'sgd': SGDClassifier,
                'mini_batch_sgd': MiniBatchSGD,
                'mlp': MLPClassifier,
                'logistic_regression': LogisticRegression,
                }

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
                'sgd': {},
                'mlp': {},
                'mini_batch_sgd': {},
                'svm': dict(verbose=True,
                            #kernel='poly',
                            #C=10, gamma=1)
                            #C=.01, gamma=.1)
                            #C=.01)
                            C=10),
                            #degree=3),
                'linear_svm': {},
                'logistic_regression': dict(C=.1),
                'grid_search': {},
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


