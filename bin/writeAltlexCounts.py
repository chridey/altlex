import sys
import json
import collections

from chnlp.utils.utils import splitData
from chnlp.utils.treeUtils import extractSelfCategory
from chnlp.altlex.config import Config
from chnlp.altlex.featureExtractor import makeDataset

with open(sys.argv[1]) as f:
    data = json.load(f)

if len(sys.argv) > 2:
    search = sys.argv[2]
else:
    search = None

config = Config()
featureSettings = config.fixedSettings
taggedData = makeDataset(data,
                         config.featureExtractor,
                         featureSettings)

training, testing = splitData(taggedData)

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
