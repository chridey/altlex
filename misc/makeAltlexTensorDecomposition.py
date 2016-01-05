import sys
import gzip
import json
import os
import time

import itertools
import collections

from chnlp.misc.extractWikipediaAltlex import iterData,PreprocessedIterator
from chnlp.ml import tensorDecomposition

class WordPairIterator:
    def __init__(self,
                 pairIterator,
                 minCount=1,
                 maxSentences=float('inf'), 
                 weighting=None, #must be a callable taking the iterator and feature as params
                 verbose=False): 

        self._pairIterator = pairIterator
        self._minCount = minCount
        self._maxSentences = maxSentences
        self._weighting = weighting
        self._verbose = verbose

        self._wordLookup = {}
        self._idLookup = {}
        self._featureCounts = None

    def _iterate(self):
        raise NotImplementedError

    def _lookupFeatureIndices(self, features):
        if features[2] not in self._idLookup:
            if len(self._idLookup) > self._maxSentences:
                return None
            self._idLookup[features[2]]  = len(self._idLookup)

        if features[0] not in self._wordLookup:
            self._wordLookup[features[0]] = len(self._wordLookup)
        if features[1] not in self._wordLookup:
            self._wordLookup[features[1]] = len(self._wordLookup)

        return self._wordLookup[features[0]], self._wordLookup[features[1]], self._idLookup[features[2]]    
    def weighting(self, feature):
        if self._weighting is None:
            return 1
        return self._weighting(self, feature)

    @property
    def shape(self):
        return len(self._wordLookup),len(self._wordLookup),len(self._idLookup)

    def __iter__(self):
        if self._featureCounts is None:
            if self._verbose:
                print('calculating feature counts at {}...'.format(time.time()))
            self._featureCounts = collections.defaultdict(int)
            for features in self._iterate():
                self._featureCounts[features[:-1]] += 1
        if self._verbose:
            print('iterating through cooccurrences at {}...'.format(time.time()))
            
        for features in self._iterate():
            if self._featureCounts[features[:-1]] < self._minCount:
                continue
            indices = self._lookupFeatureIndices(features)
            if indices is None:
                continue
            yield indices

class PairedAltlexWordPairIterator(WordPairIterator):
    def __init__(self,
                 pairIterator,
                 indices=None,
                 maxSentences = float('inf'),
                 minCount=1,
                 weighting=None,
                 verbose=False): 

        self._indices = indices
        WordPairIterator.__init__(self, pairIterator, minCount, weighting, maxSentences, verbose)
                                  
    @property
    def _iterator(self):
        return self._pairIterator.iterLabels(self._indices)
        
    def _iterate(self):
        for sentenceId,labels,altlexes,pair in self._iterator:
            if self._verbose and sentenceId % 10000 == 0:
                print(sentenceId)
            for altlexId,(datapoint, label) in enumerate(iterData(labels, altlexes, pair)):
                if label not in (0,1):
                    continue

                prev = datapoint.getPrevLemmas()
                post = datapoint.getCurrLemmasPostAltlex()
                for wordPair in itertools.product(prev, post):
                    yield wordPair[0], wordPair[1], (sentenceId, altlexId)

class PreprocessedWordPairIterator(WordPairIterator):
    def _iterate(self):
        if getattr(self, '_cooccurrences', None) is None:
            if self._verbose:
                print('loading cooccurrences at {}...'.format(time.time()))
            with gzip.open(self._pairIterator) as f:
                j = json.load(f)

            #feat_keys, feat_values, words, ids_keys, ids_values = j
            #self._featureCounts  = dict(zip(feat_keys, feat_values))
            cooc_keys, cooc_values, words, ids_keys, ids_values = j
            self._cooccurrences = cooc_keys
            self._reverseWordLookup = {j:i for i,j in words.iteritems()}
            self._reverseIdLookup = dict(zip(ids_values, ids_keys))

        for feature in self._cooccurrences:
            yield self._reverseWordLookup[feature[0]], self._reverseWordLookup[feature[1]], tuple(self._reverseIdLookup[feature[2]])
            
if __name__ == '__main__':
    parsedDataDir = sys.argv[1]
    countsFile = sys.argv[2]
    print('loading labels at {}...'.format(time.time()))
    with gzip.open(sys.argv[2]) as f:
        labels, altlexes = json.load(f)

    if os.path.exists(countsFile):
        iterator = PreprocessedWordPairIterator(sys.argv[3],
                                                minCount=1,
                                                maxSentences=1000,
                                                verbose=True)
    else:
        iterator = PairedAltlexWordPairIterator(PreprocessedIterator(parsedDataDir,
                                                                     labels,
                                                                 altlexes),
                                                minCount=1,
                                                #maxSentences=100,
                                                verbose=True)
    print('getting counts at {}...'.format(time.time()))
    cooccurrenceCounts = tensorDecomposition.build(iterator, True)
    print(iterator.shape, len(cooccurrenceCounts))
    if not os.path.exists(countsFile):
        print('saving counts...')
        with gzip.open(sys.argv[3], 'w') as f:
            json.dump([cooccurrenceCounts.keys(),
                       cooccurrenceCounts.values(),
                       iterator._wordLookup,
                       iterator._idLookup.keys(),
                       iterator._idLookup.values()],
                      f)
    rank=100
    decomposition = tensorDecomposition.decompose(cooccurrenceCounts,
                                                  iterator.shape,
                                                  rank,
                                                  30,
                                                  verbose=True)
    tensorDecomposition.save((countsFile+' ')[countsFile.find('.')],
                             decomposition,
                             rank)
