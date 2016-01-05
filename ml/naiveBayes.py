import math

from sklearn import naive_bayes

from chnlp.ml.sklearner import Sklearner

class NaiveBayes(Sklearner):
    def __init__(self):
        Sklearner.__init__(self)
        self.classifier = naive_bayes.BernoulliNB()

