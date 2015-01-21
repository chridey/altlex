from sklearn.semi_supervised import LabelSpreading

from chnlp.ml.sklearner import Sklearner

class LabelSpreader(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = LabelSpreading(kernel = 'knn')
