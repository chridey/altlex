import sys
import json
import math
from collections import defaultdict

import sklearner
from featureExtractor import FeatureExtractor
from dataPoint import DataPoint

counts = defaultdict(int)
noncounts = defaultdict(int)

with open(sys.argv[1]) as f:
    data = json.load(f)
print(len(data))

classifier = sklearner.load(sys.argv[2])


featureSets = []
import time
print(time.time())
for dataPoint in data[:1000]:
    dp = DataPoint(dataPoint)
    fe = FeatureExtractor() #dumb way of not caching
    featureSet = fe.addFeatures(dp, fe.defaultSettings)

    observed = classifier.classify(featureSet)

    a = ' '.join(dp.getAltlex())
    #print(a)
    if observed:
        #print(observed)
        counts[a] += 1
    else:
        noncounts[a] += 1

print(time.time())

for w in sorted (counts, key=lambda x:counts[x]/(counts[x]+noncounts[x])*math.log(counts[x],2)):
        print(counts[w], noncounts[w], counts[x]/(counts[x]+noncounts[x])*math.log(counts[x],2), w)
