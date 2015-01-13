import sklearn

from chnlp.ml.sklearner import Sklearner

class LogisticRegression(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = sklearn.linear_model.LogisticRegression(C=.1)

    '''
    def train(self, training, transform=True):
        X, Y = zip(*training)
        if transform:
            X = self._transform(X)
        self.model = self.classifier.fit_transform(X, Y)
        return self.model        
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
