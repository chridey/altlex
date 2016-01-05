import sys
import os
import math
import json
import time

from chnlp.utils.readers.parsedGigawordReader import ParsedGigawordReader
from chnlp.utils.readers.sentenceReader import SentenceReader

class WordPairCounter:
    def __init__(self):
        self._counter = {}
        
    def add(self, marker, word1, word2):
        if marker not in self._counter:
            self._counter[marker] = {}
        if word1 not in self._counter[marker]:
            self._counter[marker][word1] = {}
        if word2 not in self._counter[marker][word1]:
            self._counter[marker][word1][word2] = 0

        self._counter[marker][word1][word2] += 1

    def lookup(self, marker, word1, word2):
        pass

    def dump(self, fp):
        pass

    def load(self, fp):
        pass

    @property
    def counter(self):
        return self._counter
    
counter = WordPairCounter()
r = ParsedGigawordReader(sys.argv[1])

#read in discourse markers
configPath = os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'config')
with open(os.path.join(configPath, 'markers')) as f:
    markers = set(f.read().splitlines())

starttime = time.time()
#every 24 hours write out the results
timeout = 60*60*8

try:
    for s in r.iterFiles():
        sr = SentenceReader(s)

        for sentence in sr.iterSentences(False):
            for marker in markers:
                split = ' '.join(sentence.words).split(marker)
                if len(split) > 1:
                    countFirst = split[0].count(' ')
                    #countSecond = ' '.join(split[1:]).count(' ')
                    i = 0
                    while i < countFirst:
                        i,lemma1,pos1 = sentence.nextLemmaAndPOS(i,
                                                                 countFirst)
                        j = countFirst+1
                        while j < len(sentence.lemmas):
                            j,lemma2,pos2 = sentence.nextLemmaAndPOS(j,
                                                                     len(sentence.lemmas))
                            counter.add(marker, lemma1+'-'+pos1, lemma2+'-'+pos2)
                            
        if time.time() - starttime > timeout:
            with open(sys.argv[2] + '_' + str(int(time.time())) + '_counter.json', 'w') as f:
                json.dump(counter.counter, f)
            starttime = time.time()
except KeyboardInterrupt:
    pass
except Exception as e:
    print(e)

with open(sys.argv[2] + '_' + str(int(time.time())) + '_counter.json', 'w') as f:
    json.dump(counter.counter, f)

