import sys
import json
import collections

from chnlp.utils.utils import splitData
from chnlp.utils.treeUtils import extractSelfCategory
from chnlp.altlex.dataPoint import DataPoint

with open(sys.argv[1]) as f:
    data = json.load(f)

if len(sys.argv) > 2:
    search = sys.argv[2]
else:
    search = None

taggedData = []
for dataPoint in data:
    dp = DataPoint(dataPoint)
    if dp.getTag() == 'causal':
        taggedData.append((dp, True))
    else:
        taggedData.append((dp, False))
        
#training, testing = splitData(taggedData)
training = taggedData

counts = collections.defaultdict(int)
causalCounts = collections.defaultdict(int)
for dp,tag in training:
    if not search or dp.matchAltlex(search):
        counts[' '.join(dp.getAltlex())] += 1
        if tag:
            causalCounts[' '.join(dp.getAltlex())] += 1
        if search:
            print(dp.getCurrParse(), dp.getTag(),
                  extractSelfCategory(dp.getAltlex(), dp.getCurrParse()))
for s in sorted(counts, key=counts.get):
    print(s, counts[s])
    if s in causalCounts:
        print("\t{}".format(causalCounts[s]))
