import sys
import json
import collections

from chnlp.utils.utils import splitData
from chnlp.altlex.config import Config
from chnlp.altlex.featureExtractor import makeDataset
from chnlp.ml.randomizedPCA import RandomizedPCA

with open(sys.argv[1]) as f:
    data = json.load(f)

config = Config()
featureSettings = config.fixedSettings
taggedData = makeDataset(data,
                         config.featureExtractor,
                         featureSettings)

training, testing = splitData(taggedData)

rpca = RandomizedPCA()

rpca.train(training)
rpca.printResults()
