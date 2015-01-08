from sklearn.decomposition import RandomizedPCA as Rpca

from chnlp.ml.sklearner import Sklearner

class RandomizedPCA(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = Rpca(n_components=10)

    def crossvalidate(self, data):
        raise NotImplementedError

    def metrics(self, data):
        raise NotImplementedError

    def classify(self, data):
        raise NotImplementedError

    def accuracy(self, data):
        raise NotImplementedError

    def printResults(self):
        print(self.classifier.explained_variance_ratio_)

    
