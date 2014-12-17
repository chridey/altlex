import sys
import json
import collections

from utils import splitData
from treeUtils import extractSelfCategory
from dataPoint import DataPoint

with open(sys.argv[1]) as f:
    data = json.load(f)

if len(sys.argv) > 2:
    search = sys.argv[2]
else:
    search = None

noncausal = []
causal = []
for dataPoint in data:
    dp = DataPoint(dataPoint)
    
    if dp.getTag() == 'causal':
        causal.append(dp)
    else:
        noncausal.append(dp)

training, testing = splitData(causal, noncausal)

counts = collections.defaultdict(int)
causalCounts = collections.defaultdict(int)
for dp in training:
    if not search or dp.matchAltlex(search):
        counts[' '.join(dp.getAltlex())] += 1
        if dp.getTag() == 'causal':
            causalCounts[' '.join(dp.getAltlex())] += 1
        if search:
            print(dp.getCurrParse(), dp.getTag(),
                  extractSelfCategory(dp.getAltlex(), dp.getCurrParse()))
for s in sorted(counts, key=counts.get):
    print(s, counts[s])
    if s in causalCounts:
        print("\t{}".format(causalCounts[s]))
