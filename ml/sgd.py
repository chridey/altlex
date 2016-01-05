import sklearn

from chnlp.ml.sklearner import Sklearner

class SGD(Sklearner):
    def __init__(self):
        #super().__init__()
        Sklearner.__init__(self)
        self.classifier = sklearn.linear_model.SGDClassifier()

    def _feature_importances(self):
        return self.model.coef_
