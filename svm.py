from sklearn import preprocessing
from sklearn.svm import SVC
from sklearner import Sklearner

class SVM(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = SVC(verbose=True,
                              #kernel='linear',
                              C=10)
        self.scaler = preprocessing.MinMaxScaler()
        
    def _transform(self, features):
        X = super()._transform(features)
        #normalizing the data actually makes it worse??
        #return preprocessing.scale(X)
        #return self.scaler.fit_transform(X)
        return X
    
    def show_most_informative_features(self, n=50):
        #not supported for SVM
        #print out support vectors instead?
        pass
