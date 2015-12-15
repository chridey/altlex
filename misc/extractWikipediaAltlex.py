from __future__ import print_function

import gzip
import json
import os
import sys
import time
import collections

from chnlp.misc import calcKLDivergence
from chnlp.misc import wiknet

from chnlp.utils import treeUtils

from chnlp.altlex.config import Config
from chnlp.altlex.dataPoint import DataPoint

#1) a) iterate over the paired parsed data and extract potential altlexes (this should be done already)
#or b) load altlexes and parses from 1
#2) start with an initial training set using distant supervision to identify the causal datapoints (for those without a known altlex use the alignment table to figure this out, i.e. does it overlap with the known altlexes, model/aligned.grow-diag-final (both ways?))
#at training time
#for k iterations:
#3) calculate the KL divergence for every possible altlex but only causal/non-causal wordpairs (word pairs only for split sentences? no, not yet), also only for sentences that are all causal or all-noncausal or other
#4) do a matrix or tensor factorization on delta KLD weighted wordpairs TODO
#5) split each sentence on the possible altlex creating a new datapoint for each
#6) extract any other features, lexical, binary, etc.
#6a) balance the data
#7) classify against unknown data, only add them to training if both pairs agree (the identified altlexes must be in each other's phrase tables)
#8) go to step 3

featureSettings = {'altlex_length': True,
                   'final_reporting': True,
                   'final_time': True,
                   'head_verb_altlex' : True,
                   'head_verb_curr' : True,
                   'head_verb_prev' : True,
                   'verbnet_class_prev' : True,
                   'verbnet_class_curr' : True, #both seem to help
                   'framenet' : True}                  

def getMetaLabel(labels):
    if len(labels) == 0:
        return 'other'
    elif not(any(k[0] for k in labels)):
        return 'notcausal'
    elif all(k[0] for k in labels):
        return 'causal'
    return 'both'

class PreprocessedIterator(calcKLDivergence.ParsedPairIterator):
    def __init__(self, indir, labels, altlexes, verbose=False, numCausal=100, numNonCausal=1000,
                 ignoreBoth=False): 
        self.indir = indir
        self.labels = labels
        self.altlexes = altlexes
        self.verbose = verbose
        self.numCausal = numCausal
        self.numNonCausal = numNonCausal
        self.ignoreBoth = ignoreBoth

    def iterLabels(self):
        for index,pair in enumerate(calcKLDivergence.ParsedPairIterator.__iter__(self)):
            labels = self.labels[index]
            altlexes = self.altlexes[index]
            yield labels,altlexes,pair

    def iterAltlexes(self):
        for labels,altlexes,pair in self.iterLabels():
            lemmas = [[i.lower() for i in wiknet.getLemmas(pair[0]['lemmas'])],
                      [i.lower() for i in wiknet.getLemmas(pair[1]['lemmas'])]]
            pos = [wiknet.getLemmas(pair[0]['pos']),
                   wiknet.getLemmas(pair[1]['pos'])]

            '''
            this is what each altlex looks like
            alignedAltlex = [[start,
                              start+len(altlexes[pairIndex][startIndex])//2],
                             [startList[1-pairIndex][match],
                             startList[1-pairIndex][match]+len(altlexes[1-pairIndex][match])//2]]
            '''
            #first get all the ngrams
            stems = [[], []]
            for a in altlexes:
                stems[0].append(tuple(lemmas[0][a[0][0]:a[0][1]] + pos[0][a[0][0]:a[0][1]]))
                stems[1].append(tuple(lemmas[1][a[1][0]:a[1][1]] + pos[1][a[1][0]:a[1][1]]))
            yield labels,stems
        
    def __iter__(self):
        causalCount = 0
        nonCausalCount = 0
        for labels,stems in self.iterAltlexes():
            stems = [set(stems[0]), set(stems[1])]
            #then determine whether this is a training point and what type it is
            metaLabel = getMetaLabel(labels)

            if metaLabel == 'notcausal':
                nonCausalCount += 1
                if nonCausalCount <= self.numNonCausal:
                    continue
            elif metaLabel == 'causal':
                causalCount += 1
                if causalCount <= self.numCausal:
                    continue
            elif metaLabel == 'both' and self.ignoreBoth:
                continue
            
            yield metaLabel,stems[0],stems[1]
            yield metaLabel,stems[1],stems[0]

    def getIndices(self):
        indices = collections.defaultdict(list)
        for index,(labels,altlexes,pair) in enumerate(pairIterator.iterLabels()):
            metaLabel = getMetaLabel(labels)
            indices[metaLabel].append(index)
        return indices

    def getTestData(self):
        causalCount, nonCausalCount = 0, 0
        test = []
        for index,(labels,altlexes,pair) in enumerate(self.iterLabels()):
            if index % 10000 == 0:
                print(index)
            metaLabel = getMetaLabel(labels)

            if metaLabel == 'causal':
                if causalCount >= self.numCausal:
                    continue
                causalCount += 1
            elif metaLabel == 'notcausal':
                if nonCausalCount >= self.numNonCausal:
                    continue
                nonCausalCount += 1
            else:
                continue

            labelLookup = {j:i for i,j in labels}
            lemmas = [[i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[0]['lemmas'])],
                      [i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[1]['lemmas'])]]
            pos = [wiknet.getLemmas(pair[0]['pos']),
                   wiknet.getLemmas(pair[1]['pos'])]
            for index,altlex in enumerate(altlexes):
                for i,which in enumerate(altlex):
                    altlex_ngram = tuple(lemmas[i][which[0]:which[1]])
                    altlex_pos = tuple(pos[i][which[0]:which[1]])
                    label = labelLookup.get(index, 2)
                    test.append((altlex_ngram+altlex_pos, label))

            if causalCount >= self.numCausal and nonCausalCount >= self.numNonCausal:
                break

        return test
                    
def getAlignedAltlexes(alignment, lemmas, altlexes):
    alignedAltlexes = []
    aligned = [{}, {}]
    startList = [[], []]

    for pairIndex,ngrams in enumerate(altlexes):
        for ngramIndex,ngram in enumerate(ngrams):
            #figure out where this ngram occurs in the actual data
            start = treeUtils.findPhrase(ngram[:len(ngram)//2],
                                         lemmas[pairIndex])
            startList[pairIndex].append(start)
            if start is None:
                print("Problem finding phrase {} IN {}".format(ngram, lemmas[pairIndex]),
                      file=sys.stderr)
                continue
            for i in range(start, start+len(ngram)//2):
                aligned[pairIndex][i] = ngramIndex

    if lemmas[0] != lemmas[1]:
        print(altlexes, startList, aligned)
    #now go through each altlex and see if at least part of it has an alignment or part of it aligns to a period (TODO)
    alreadyFound = set()
    alignedAltlexes = []
    for pairIndex,starts in enumerate(startList):
        for startIndex,start in enumerate(starts):
            if start is None or (pairIndex,startIndex) in alreadyFound:
                continue
            match = None
            for i in range(start, start+len(altlexes[pairIndex][startIndex])//2):
                if i in alignment[pairIndex] and alignment[pairIndex][i] in aligned[1-pairIndex]:
                    match = aligned[1-pairIndex][alignment[pairIndex][i]]
                    break
            if match is not None:
                alignedAltlex = [[start,
                                  start+len(altlexes[pairIndex][startIndex])//2],
                                 [startList[1-pairIndex][match],
                                  startList[1-pairIndex][match]+len(altlexes[1-pairIndex][match])//2]]
                if not pairIndex:
                    alignedAltlexes.append(alignedAltlex)
                else:
                    alignedAltlexes.append(alignedAltlex[::-1])
                alreadyFound.add((pairIndex, startIndex))
                alreadyFound.add((1-pairIndex, match))
    return alignedAltlexes

def makeDataset(pairIterator,
                deltaKLD,
                causalPhrases,
                featureExtractor,
                numCausal,
                numNonCausal,
                precalculated=None):

    causalCount = 0
    nonCausalCount = 0
    total = 0
    if precalculated is None:
        train, test, unclassified = [], [], []
    else:
        train, test, unclassified = precalculated
    indices = [0, 0, 0]
    datasets = [train, test, unclassified]
    for labels,altlexes,pair in pairIterator.iterLabels():
        total += 1
        if total % 10000 == 0:
            print(total)
        metaLabel = getMetaLabel(labels)
        currentIndex = None
        if metaLabel == 'causal':
            if causalCount < numCausal:
                currentIndex = 1
            causalCount += 1
        elif metaLabel == 'notcausal':
            if nonCausalCount < numNonCausal:
                currentIndex = 1
            nonCausalCount += 1
        elif metaLabel == 'other':
            currentIndex = 2
            
        if currentIndex is None:
            currentIndex = 0
        
        labelLookup = {j:i for i,j in labels}
        lemmas = [[i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[0]['lemmas'])],
                  [i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[1]['lemmas'])]]        
        pos = [wiknet.getLemmas(pair[0]['pos']),
               wiknet.getLemmas(pair[1]['pos'])]
        for index,altlex in enumerate(altlexes):
            for i,which in enumerate(altlex):
                altlex_ngram = tuple(lemmas[i][which[0]:which[1]])
                altlex_pos = tuple(pos[i][which[0]:which[1]])
                if precalculated is None:
                    dataPoint = {'altlexLength': which[1]-which[0],
                                 'sentences': [{
                                     'lemmas': lemmas[i][which[0]:],
                                     'words': lemmas[i][which[0]:],
                                     'stems': lemmas[i][which[0]:],
                                     'pos': pos[i][which[0]:]
                                     },
                                     {
                                     'lemmas': lemmas[i][:which[0]],
                                     'words': lemmas[i][:which[0]],
                                     'stems': lemmas[i][:which[0]],
                                      'pos': pos[i][:which[0]]
                                     }]}

                    dp = DataPoint(dataPoint)
                    features = featureExtractor.addFeatures(dp, featureSettings)

                else:
                    features = datasets[currentIndex][indices[currentIndex]][0]
                    
                    #print(altlex_ngram+altlex_pos)
                features['causal_kld'] = deltaKLD['causal'].get(altlex_ngram + altlex_pos, 0)
                features['notcausal_kld'] = deltaKLD['notcausal'].get(altlex_ngram + altlex_pos, 0)
                features['other_kld'] = deltaKLD['other'].get(altlex_ngram + altlex_pos, 0)
                #print(features)
                features['in_causal'] = altlex_ngram in causalPhrases['causal']
                features['in_notcausal'] = altlex_ngram in causalPhrases['notcausal']
                #TODO: add other features

                label = labelLookup.get(index, 2)
                if precalculated:
                    datasets[currentIndex][indices[currentIndex]] = (features, label)
                else:
                    datasets[currentIndex].append((features, label))
                indices[currentIndex] += 1

    return train, test, unclassified

if __name__ == '__main__':

    pairIterator = calcKLDivergence.ParsedPairIterator(sys.argv[1])
    #print('loading potential altlexes...')
    #with gzip.open(sys.argv[2]) as f:
    #    potentialAltlexes = json.load(f)
    #print(len(potentialAltlexes))
    print('loading alignments...')
    with open(sys.argv[2]) as f:
        alignments = f.read().splitlines()
    print('loading phrases...')
    with gzip.open(sys.argv[3]) as f:
        phrases = json.load(f) 

    prefix = sys.argv[4]

    seedSet = {'causal': [set(i) for i in calcKLDivergence.causal_markers],
               'notcausal': [set(i) for i in calcKLDivergence.noncausal_markers]}

    #if file of initial labels does not exist
    if not os.path.exists('initLabels.json.gz'):
        labels = [] #list of tuples(label, altlexIndex)
        altlexes = [] #list of list of tuples ((sent0Start, sent0End), (sent1Start, sent1End)) ...
        labelLookup = {0 : 'notcausal',
                       1 : 'causal',
                       2 : 'other'}               

        for index,pair in enumerate(pairIterator):
            alignment = [{int(i.split('-')[1]):int(i.split('-')[0]) for i in alignments[index].split()}]
            alignment.append({j:i for i,j in alignment[0].items()})
            lemmas = [[i.lower() for i in wiknet.getLemmas(pair[0]['lemmas'])],
                      [i.lower() for i in wiknet.getLemmas(pair[1]['lemmas'])]]
            pos = [wiknet.getLemmas(pair[0]['pos']),
                   wiknet.getLemmas(pair[1]['pos'])]
            potentialAltlexes = calcKLDivergence.getNgrams(pair)
            alignedAltlexes = getAlignedAltlexes(alignment, lemmas, potentialAltlexes)

            labelInfo = []
            for altlexIndex,alignedAltlex in enumerate(alignedAltlexes):
                label = None
                for causalType in range(2):
                    for i,altlex in enumerate(alignedAltlex):
                        if any(seed.issubset(lemmas[i][altlex[0]:altlex[1]] + pos[i][altlex[0]:altlex[1]]) for seed in seedSet[labelLookup[causalType]]):
                            label = causalType
                if label is not None:
                    labelInfo.append([label, altlexIndex])
            labels.append(labelInfo)
            altlexes.append(alignedAltlexes)

            #TODO - also handle multi sentences - this should work fine i think?
            if pair[0]['lemmas']  != pair[1]['lemmas']:
                print(pair[0]['lemmas'], pair[1]['lemmas'], labelInfo, alignedAltlexes, alignment)

        #save labels AND aligned altlexes
        #labels will change at each iteration, aligned altlexes will not
        with gzip.open('initLabels.json.gz', 'w') as f:
            json.dump([labels, altlexes], f)
    else:
        print('loading labels...')
        with gzip.open('initLabels.json.gz') as f:
            labels, altlexes = json.load(f)

    k = 1
    numCausalTesting = 100
    numNonCausalTesting = 1000
    config = Config()
    if os.path.exists('features.json.gz'):
        with gzip.open('features.json.gz') as f:
            precalculated = json.load(f)
    else:
        precalculated = None
        
    for i in range(k):
        #calculate KL divergence
        print(i)
        pairIterator = PreprocessedIterator(sys.argv[1], labels, altlexes, verbose=False, ignoreBoth=True, numCausal=numCausalTesting, numNonCausal=numNonCausalTesting)
        print('calculating kld at {}...'.format(time.time()))
        kldt = calcKLDivergence.main(pairIterator, withS1=False, prefix=prefix + str(i))    
        deltaKLD = {'causal': {},
                    'notcausal': {},
                    'other': {}}
        for phraseType in 'causal', 'notcausal', 'other':
            topKLD = kldt[phraseType][1].topKLD()
            for kld in topKLD:
                if kld[1] > kld[2]:
                    score = kld[3]
                else:
                    score = -kld[3]
                deltaKLD[phraseType][kld[0]] = score
        for q in deltaKLD:
            print(len(deltaKLD[q]))
            
        #calculate causal phrases from starting seeds
        print('calculating causal mappings at {}...'.format(time.time()))
        causalPhrases = calcKLDivergence.getCausalPhrases(phrases['phrases'], seedSet, stem=False)
        #TODO: some kind of factorization
        #add features
        print('adding features at {}...'.format(time.time()))
        train,test,unclassified = makeDataset(pairIterator,
                                              deltaKLD,
                                              causalPhrases,
                                              config.featureExtractor,
                                              numCausal=numCausalTesting,
                                              numNonCausal=numNonCausalTesting,
                                              precalculated=precalculated)
        print(time.time())
        print(len(train))
        print(len(test))
        print(len(unclassified))

        with gzip.open('features.json.gz', 'w') as f:
            json.dump([train, test, unclassified], f)
        #train classifier on marked data points
        #add new datapoints to labels
