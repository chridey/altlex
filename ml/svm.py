from sklearn import preprocessing
from sklearn.svm import SVC

from chnlp.ml.sklearner import Sklearner

class SVM(Sklearner):
    def __init__(self):
        #super().__init__()
        Sklearner.__init__(self)
        self.classifier = SVC(verbose=True,
                              #kernel='poly',
                              #C=10, gamma=1)
                              #C=.01, gamma=.1)
                              #C=.01)
                              C=10)
                              #degree=3)
        self.scaler = preprocessing.MinMaxScaler()
        
    def _transform(self, features):
        #X = super()._transform(features)
        X = Sklearner._transform(self, features)
        #normalizing the data actually makes it worse??
        #if len(X):
        #    pass
            #return preprocessing.scale(X)
            #return self.scaler.fit_transform(X)
        return X
    
    def show_most_informative_features(self, n=50):
        #not supported for SVM
        #print out support vectors instead?
        pass

    def confidence(self, features, transform=True):
        if transform:
            assert(type(features) == dict)
            X = self._transform([features])
        else:
            X = features
            
        result = abs(self.model.decision_function(X))

        return result[0]
