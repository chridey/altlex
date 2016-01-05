from sklearn.cluster import AffinityPropagation

from chnlp.ml.sklearner import Sklearner

class AffinityPropagator(Sklearner):
    def __init__(self):
        Sklearner.__init__(self)
        #super().__init__()
        self.classifier = AffinityPropagation(verbose=True)

    def train(self, X, transform=True):
        if transform:
            X = self._transform(X)
        self.model = self.classifier.fit(X)
        return self.model
    
