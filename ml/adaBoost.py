from sklearn.ensemble import AdaBoostClassifier

from chnlp.ml.sklearner import Sklearner

class AdaBoost(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = AdaBoostClassifier(n_estimators=500)
        
