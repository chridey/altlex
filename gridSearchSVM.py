from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from svm import SVM

class GridSearchSVM(SVM):
    def __init__(self):
        super().__init__()
        svc = SVC(verbose=True)
        self.parameters = {'C': (.01, .1, 1, 10),
                           'gamma': (.01, .1, 1, 10)}
        self.classifier = GridSearchCV(svc, self.parameters)

    def train(self, features, transform=True):
        super().train(features, transform)
        print(self.classifier.best_params_)
        self.model = self.classifier.best_estimator_
        return self.model
        
    def show_most_informative_features(self, n=50):
        #not supported for SVM
        #print out support vectors instead?
        pass
