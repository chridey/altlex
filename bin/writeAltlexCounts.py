import sys
import json
import argparse

import collections

from chnlp.utils.utils import splitData
from chnlp.utils.treeUtils import extractSelfCategory
from chnlp.altlex.dataPoint import DataPoint

parser = argparse.ArgumentParser(description='explore a dataset with altlexes')

parser.add_argument('infile', 
                    help='the file containing the sentences and metadata in JSON format')
parser.add_argument('--search', '-s', metavar='S',
                    help = 'search for a specific altlex')

parser.add_argument('--all', '-a', action='store_true',
                    help = 'use all the data')

parser.add_argument('--printSentence', '-p', action='store_true',
                    help = 'print out the sentences for each data point')

args = parser.parse_args()

with open(args.infile) as f:
    data = json.load(f)
search = args.search

taggedData = []
for dataPoint in data:
    dp = DataPoint(dataPoint)
    if dp.getTag() == 'causal':
        taggedData.append((dp, True))
    else:
        taggedData.append((dp, False))

if args.all:
    training = taggedData
else:
    training, testing = splitData(taggedData)
    print(len(testing))

counts = collections.defaultdict(int)
causalCounts = collections.defaultdict(int)
altlexLookup = collections.defaultdict(list)

print('Total: {}'.format(len(training)))
for dp,tag in training:
    if not search or dp.matchAltlex(search):
        altlex = ' '.join(dp.getAltlex())
        counts[altlex] += 1
        if args.printSentence:
            sentences = dp.getSentences()
            altlexLookup[altlex].append(sentences)
        if tag:
            causalCounts[altlex] += 1
        if search:
            print(dp.getCurrParse(), dp.getTag(),
                  extractSelfCategory(dp.getAltlex(), dp.getCurrParse()))

for s in sorted(counts, key=counts.get):
    print(s, counts[s])
    if s in causalCounts:
        print("\t{}".format(causalCounts[s]))
    if s in altlexLookup:
        for i in altlexLookup[s]:
            print('\t', i)

