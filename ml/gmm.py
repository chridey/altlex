from sklearn.mixture import GMM as SKGMM

from chnlp.ml.sklearner import Sklearner

class GMM(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = SKGMM(n_components=2,
                                n_iter=100)

    def train(self, training, transform=True):
        X, Y = zip(*training)
        if transform:
            X = self._transform(X)
        self.model = self.classifier.fit(X)
        return self.model

    def accuracy(self, testing, transform=True):
        return 0

    def classify(self, features, transform=True):
        if transform:
            assert(type(features) == dict)
            X = self._transform([features])
        else:
            X = features

        #X is a 1D array of shape (num_features)
        #but it needs to be a 2D array of shape (1, num_features)
        result = self.model.predict([X])
        return result[0]

    def show_most_informative_features(self, n=50):
        pass



