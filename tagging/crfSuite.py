#read in all sentences in json format
#ignore the ones that start with an explicit discourse marker (actually, keep them?)
#only output the second sentence, but each word should be marked with features
#possible tags:
#B-CAL (begin causal altlex)
#I-CAL
#B-AL (begin altlex)
#I-AL
#B-DM (explicit marker)
#I-DM
#O (everything else)

from chnlp.ml.sklearner import Sklearner

class CRFSuiteLearner(Sklearner):
    def _transform(self, features):
        
