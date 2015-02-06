import sys
import os
import math
import json
import time

from collections import defaultdict, Counter

import numpy

from nltk.stem import SnowballStemmer

from chnlp.utils.readers.parsedGigawordReader import ParsedGigawordReader
from chnlp.utils.readers.sentenceReader import SentenceReader

sn = SnowballStemmer('english')

with open(sys.argv[1]) as f:
    seedWords = {sn.stem(i) for i in f.read().splitlines()}

r = ParsedGigawordReader(sys.argv[2])

seedLookup = {j:i for (i,j) in enumerate(seedWords)}
numSeeds = len(list(seedLookup))
wordLookup = {}
counts = []

idf = Counter()
N = 0
sentence_idf = Counter()
sentenceN = 0

starttime = time.time()
timeout = 60*60*57
try:
    for s in r.iterFiles():
        sr = SentenceReader(s)

        N += 1
        docStems = set()
        for sentence in sr.iterSentences(False):
            stems = set(sentence.stems)
            docStems.update(stems)
            sentenceN += 1
            sentence_idf.update(stems)

            intersection = seedWords & stems
            #if intersection:
            for stem in sentence.stems:
                if stem not in wordLookup:
                    wordLookup[stem] = len(counts)
                    counts.append([0 for i in range(numSeeds)])
                for seed in intersection:
                    counts[wordLookup[stem]][seedLookup[seed]] += 1

        idf.update(docStems)
        
        if time.time() - starttime > timeout:
            raise Exception

except KeyboardInterrupt:
    pass
except Exception:
    pass

'''
reverseWordLookup = {j:i for (i,j) in wordLookup.items()}
#multiply by IDF then save
for wordIndex in range(len(counts)):
    for seedIndex in range(len(counts[wordIndex])):
        counts[wordIndex][seedIndex] * math.log(N/(idf[reverseWordLookup[wordIndex]]))
'''

numpy.array(counts).dump(sys.argv[3])

lookup = {'words': wordLookup,
          'seeds': seedLookup,
          'sentence_idf': sentence_idf,
          'sentenceN': sentenceN,
          'idf': idf,
          'N': N}

with open(sys.argv[3] + '_lookup.json', 'w') as f:
    json.dump(lookup, f)

