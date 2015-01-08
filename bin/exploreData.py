import sys
import json
import collections

from chnlp.utils.utils import splitData
from chnlp.altlex.dataPoint import DataPoint
from chnlp.altlex.featureExtractor import FeatureExtractor
from chnlp.ml.randomizedPCA import RandomizedPCA

with open(sys.argv[1]) as f:
    data = json.load(f)

noncausal = []
causal = []
fe = FeatureExtractor()
for dataPoint in data:
    dp = DataPoint(dataPoint)
    features = fe.addFeatures(dp, fe.defaultSettings)
    
    if dp.getTag() == 'causal':
        causal.append((features,True))
    else:
        noncausal.append((features,False))

training, testing = splitData(causal, noncausal)

rpca = RandomizedPCA()

rpca.train(training)
rpca.printResults()
