import sys
import json
import collections
import argparse

import matplotlib.pyplot as plt

from chnlp.utils.utils import splitData
from chnlp.altlex.config import Config
from chnlp.altlex.featureExtractor import makeDataset
from chnlp.ml.randomizedPCA import RandomizedPCA

parser = argparse.ArgumentParser(description='train and/or evaluate a classifier on a dataset with altlexes')

parser.add_argument('infile', 
                    help='the file containing the sentences and metadata in JSON format')

parser.add_argument('--feature',
                    help = 'Examine a specific feature')

parser.add_argument('--plot', metavar='P',
                    help = 'Plot a specific numeric feature and save to file P')

parser.add_argument('--featureSubset',
                    help = 'Examine a specific subset of features')

parser.add_argument('--config',
                    help = 'JSON config file to use instead of command line options')

parser.add_argument('--randomizedPCA', action='store_true',
                    help = 'see which features are most important by randomized PCA')

args = parser.parse_args()

with open(args.infile) as f:
    data = json.load(f)

if args.config:
    with open(args.config) as cf:
        settings = json.load(cf)
        config = Config(settings)
else:
    config = Config()

if args.featureSubset:
    featureSettings = config.featureSubset(args.featureSubset)
elif args.feature:
    featureSettings = {args.feature : True}
else:
    featureSettings = config.fixedSettings
    
taggedData = makeDataset(data,
                         config.featureExtractor,
                         featureSettings)

training, testing = splitData(taggedData)

if args.plot:
    features, labels = zip(*training)
    values = [f[args.feature] if args.feature in f else 0 for f in features]
    #print(values)
    plt.plot(sorted(values))
    plt.savefig(args.plot)

if args.randomizedPCA:
    rpca = RandomizedPCA()

    rpca.train(training)
    rpca.printResults()
