from sklearn.feature_extraction.text import TfidfTransformer

from chnlp.ml.sklearner import Sklearner

class TfIdf(Sklearner):
    def __init__(self):
        #super().__init__()
        Sklearner.__init__(self)
        self.classifier = TfidfTransformer()
