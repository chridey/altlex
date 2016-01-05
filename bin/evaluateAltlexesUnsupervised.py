import sys
import json
import math
from collections import defaultdict
import time

import chnlp.ml.sklearner as sklearner
from chnlp.altlex.featureExtractor import FeatureExtractor
from chnlp.altlex.dataPoint import DataPoint

counts = defaultdict(int)
noncounts = defaultdict(int)

print(time.time())
with open(sys.argv[1]) as f:
    data = json.load(f)
print(time.time())
print(len(data))

classifier = sklearner.load(sys.argv[2])

featureSets = []

print(time.time())
for dataPoint in data:
    dp = DataPoint(dataPoint)
    fe = FeatureExtractor() #dumb way of not caching
    featureSet = fe.addFeatures(dp, fe.defaultSettings)
    observed = classifier.classify(featureSet)
    print(dp.getAltlex(), featureSet, observed)
    a = ' '.join(dp.getAltlex())
    #print(a)
    if observed:
        #print(observed)
        counts[a] += 1
    else:
        noncounts[a] += 1

print(time.time())

for w in sorted (counts, key=lambda x:1.0*counts[x]/(counts[x]+noncounts[x])*math.log(counts[x],2), reverse=True):
        print(counts[w], noncounts[w], 1.0*counts[w]/(counts[w]+noncounts[w])*math.log(counts[w],2), w)
