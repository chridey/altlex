from __future__ import print_function

import sys
import os
import collections

from chnlp.word2vec import sentenceRepresentation

wikiFilename = sys.argv[1]
categoryDirName = sys.argv[2]
modelFilename = None
sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiFilename,
                                                                modelFilename)

titles = {j:i for i,j in enumerate(sentRep.titles)}

#look for files that appear in both english and simple
#for now just write article name and number of sentences and num with markers

stats = {i:collections.defaultdict(int) for i in ('sum0',
                                                  'total0',
                                                  'sum1',
                                                  'total1')}

filenames = set()
for filename in os.listdir(categoryDirName):
    filenames.add(filename)
    print(filename, file=sys.stderr)
    with open(os.path.join(categoryDirName,filename)) as f:
        for line in f:
            try:
                i,depth,category,title = line.strip().split("\t")
            except ValueError:
                print('Problem with {}'.format(line), file=sys.stderr)
                continue
            if title in titles:
                sent0 = sentRep.lookupSentences(titles[title], 0)
                sent1 = sentRep.lookupSentences(titles[title], 1)
                numCausal0 = sum('because' in i for i in sent0)
                numCausal1 = sum('because' in i for i in sent1)
                print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(title,
                                                      category,
                                                      filename,
                                                      len(sent0),
                                                      len(sent1),
                                                      numCausal0,
                                                      numCausal1))
                for cat in (filename, category):
                    stats['total0'][cat] += 1
                    stats['total1'][cat] += 1
                    stats['sum0'][cat] += 1.*numCausal0
                    stats['sum1'][cat] += 1.*numCausal1

                #import random
                #if random.random() > .5:
                #    break
                
print('-'*79, file=sys.stderr)
print('AVERAGES:', file=sys.stderr)
print('FILES:', file=sys.stderr)
for filename in filenames:
    print("\t".join(['{}']*5).format(filename,
                                     stats['total0'][filename],
                                     stats['sum0'][filename] / stats['total0'][filename],
                                     stats['sum1'][filename] / stats['total1'][filename],
                                     (stats['sum0'][filename] + stats['sum1'][filename]) / (stats['total0'][filename] + stats['total1'][filename])), file=sys.stderr)
print('CATEGORIES:', file=sys.stderr)                                     
for category in sorted(stats['total0'], key=lambda x:stats['total0'][x], reverse=True):
    if category in filenames:
        continue
    print("\t".join(['{}']*5).format(category,
                                     stats['total0'][category],
                                     stats['sum0'][category] / stats['total0'][category],
                                     stats['sum1'][category] / stats['total1'][category],
                                     (stats['sum0'][category] + stats['sum1'][category]) / (stats['total0'][category] + stats['total1'][category])), file=sys.stderr)
                                     
