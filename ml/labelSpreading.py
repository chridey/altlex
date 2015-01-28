from sklearn.semi_supervised import LabelSpreading,LabelPropagation

from chnlp.ml.sklearner import Sklearner

class LabelSpreader(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = LabelSpreading(kernel = 'knn', alpha=1)

    def show_most_informative_features(self, *args, **kwargs):
        pass

class LabelPropagator(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = LabelPropagation(kernel = 'knn', alpha=1)
