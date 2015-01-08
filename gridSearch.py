from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as lr
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

from svm import SVM
from logisticRegression import LogisticRegression
from sklearner import Sklearner

class GridSearch(Sklearner):
    def setClassifier(self):
        #skf = StratifiedKFold(y,
        #                      n_folds=2)

        self.classifier = GridSearchCV(self.classifierType,
                                       self.parameters,
                                       scoring = 'f1',
                                       cv=2)
                                       #cv=skf)
    
    def train(self, features, transform=True):
        super().train(features, transform)
        print(self.classifier.best_params_)
        self.model = self.classifier.best_estimator_
        return self.model
    
class GridSearchSVM(SVM, GridSearch):
    def __init__(self):
        super().__init__()
        self.classifierType = SVC()#(verbose=True)
        self.parameters = {'C': (.01, .1, 1, 10, 100),
                           'gamma': (0, .0001, .001, .005, .01, .1, 1)}
        self.setClassifier()
        
    def show_most_informative_features(self, n=50):
        #not supported for SVM
        #print out support vectors instead?
        pass

class GridSearchLogit(LogisticRegression, GridSearch):
    def __init__(self):
        super().__init__()
        self.classifierType = lr()
        self.parameters = {'C': (.01, .1, 1, 10, 100)}
        self.setClassifier()
        
    def show_most_informative_features(self, n=50):
        #TODO
        pass
    
