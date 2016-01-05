from sklearn.ensemble import RandomForestClassifier

from chnlp.ml.sklearner import Sklearner

class RandomForest(Sklearner):
    def __init__(self):
        #super().__init__()
        Sklearner.__init__(self)
        self.classifier = RandomForestClassifier(n_estimators=1000,
                                                 #tried 10, 100, and 500
                                                 #higher number increases precision
                                                 n_jobs=1,
                                                 oob_score=False,
                                                 bootstrap=True,
                                                 #False reduces recall
                                                 max_features="sqrt",
                                                 #using all reduces precision
                                                 #log2 reduces recall w/o p incr
                                                 #.5 reduces precision (lt max tho)
                                                 min_samples_split=2
                                                 #1 and 2 no diff
                                                 #3 is worse
                                                 )
                
        
