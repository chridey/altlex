from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading,LabelPropagation

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold



from chnlp.ml.sgd import SGD
from chnlp.ml.labelSpreading import LabelSpreader,LabelPropagator
from chnlp.ml.sklearner import Sklearner

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.base import BaseEstimator

class GridSearch(BaseEstimator):
    def setClassifier(self):
        #skf = StratifiedKFold(y,
        #                      n_folds=2)

        self.classifier = GridSearchCV(self.classifierType(**self.parameters),
                                       self.searchParameters,
                                       scoring = 'f1',
                                       cv=4,
                                       n_jobs=4,
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
        return self.classifier.fit(X, y)
    
class GridSearchSVM(GridSearch):
    def __init__(self):
        #super().__init__()
        GridSearch.__init__(self)
        self.classifierType = SVC#(kernel='linear')#(verbose=True)
        self.parameters = {}#'kernel': 'linear'}
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

class GridSearchSGD(GridSearch, SGD):
    def __init__(self):
        #super().__init__()
        GridSearch.__init__(self)
        self.classifierType = SGDClassifier
        self.parameters = {}
        self.searchParameters = {'penalty': ('l2', 'elasticnet'),
                           'alpha': (0.001, 0.0001, 0.00001)}
        self.setClassifier()
        
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
                 
