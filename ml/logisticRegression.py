from sklearn import preprocessing
import sklearn

from chnlp.ml.sklearner import Sklearner

class LogisticRegression(Sklearner):
    def __init__(self):
        #super().__init__()
        Sklearner.__init__(self)
        self.classifier = sklearn.linear_model.LogisticRegression(C=.1)
        self.scaler = preprocessing.MinMaxScaler()

    '''
    def train(self, training, transform=True):
        X, Y = zip(*training)
        if transform:
            X = self._transform(X)
        self.model = self.classifier.fit_transform(X, Y)
        return self.model        
    '''

    '''
    def _transform(self, features):
        X = super()._transform(features)
        #return preprocessing.scale(X)
        #return self.scaler.fit_transform(X)
        return X
    '''
    
    @property
    def _feature_importances(self):
        return [abs(i) for i in self.model.coef_[0]]

    def prob(self, features, transform=True):
        if transform:
            assert(type(features) == dict)
            X = self._transform([features])
        else:
            X = features
            
        result = self.model.predict_proba(X)

        return result[0]

    def confidence(self, features, transform=True):
        return self.prob(features, transform)
