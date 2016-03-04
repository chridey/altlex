from __future__ import print_function

import gzip
import json
import collections

from altlex.utils import wordUtils
from altlex.utils import dependencyUtils
from altlex.utils.readers.parsedPairIterator import ParsedPairIterator, getLemmas, getPossibleAltlexes
from altlex.featureExtraction.dataPoint import DataPoint

def getAlignedAltlexes(alignment, lemmas, altlexes, verbose=False):
    alignedAltlexes = []
    aligned = [{}, {}]
    startList = [[], []]

    for pairIndex,ngrams in enumerate(altlexes):
        for ngramIndex,ngram in enumerate(ngrams):
            #figure out where this ngram occurs in the actual data
            start = wordUtils.findPhrase(ngram[:len(ngram)//2],
                                         lemmas[pairIndex])
            startList[pairIndex].append(start)
            if start is None:
                if verbose:
                    print("Problem finding phrase {} IN {}".format(ngram, lemmas[pairIndex]),
                          file=sys.stderr)
                continue
            for i in range(start, start+len(ngram)//2):
                aligned[pairIndex][i] = ngramIndex

    if lemmas[0] != lemmas[1]:
        if verbose:
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


class AlignedParsedPairIterator(ParsedPairIterator):
    def __init__(self, indir, alignments, verbose=False):
        ParsedPairIterator.__init__(self, indir, verbose)
        self.alignments = alignments

        #list of tuples(label, altlexIndex)
        self.labels = []

        #list of list of tuples ((sent0Start, sent0End), (sent1Start, sent1End)) 
        self.altlexes = []  

        self.labelLookup = {}

    @property
    def numSentences(self):
        return len(self.altlexes)
        
    def makeLabels(self, seedSet, labelLookup):
        self.labelLookup = labelLookup
        self.reverseLabelLookup = {j:i for i,j in self.labelLookup.items()}
        
        for index,pair in enumerate(self.__iter__()):
            if self.verbose:
                print(index)

            alignment = [{int(i.split('-')[1]):int(i.split('-')[0]) for i in self.alignments[index].split()}]
            alignment.append({j:i for i,j in alignment[0].items()})
            
            lemmas = [[i.lower() for i in getLemmas(pair[0]['lemmas'])],
                      [i.lower() for i in getLemmas(pair[1]['lemmas'])]]
            pos = [getLemmas(pair[0]['pos']),
                   getLemmas(pair[1]['pos'])]

            potentialAltlexes = getPossibleAltlexes(pair)
            alignedAltlexes = getAlignedAltlexes(alignment,
                                                 lemmas,
                                                 potentialAltlexes)

            labelInfo = []
            for altlexIndex,alignedAltlex in enumerate(alignedAltlexes):
                label = None
                for causalType in seedSet.keys():
                    for i,altlex in enumerate(alignedAltlex):
                        if any(seed.issubset(lemmas[i][altlex[0]:altlex[1]] + pos[i][altlex[0]:altlex[1]]) for seed in seedSet[causalType]):
                            label = labelLookup[causalType]
                if label is not None:
                    labelInfo.append([label, altlexIndex])

            self.labels.append(labelInfo)
            self.altlexes.append(alignedAltlexes)

            #TODO - also handle multi sentences - this should work fine i think?
            if pair[0]['lemmas']  != pair[1]['lemmas']:
                if self.verbose:
                    print(pair[0]['lemmas'],
                          pair[1]['lemmas'],
                          labelInfo,
                          alignedAltlexes,
                          alignment)

    def iterLabeledPairs(self, sentenceIndices=None):
        for index,pair in enumerate(self.__iter__()):
            if sentenceIndices is not None and index not in sentenceIndices:
                continue
            labels = self.labels[index]
            altlexes = self.altlexes[index]
            yield index,labels,altlexes,pair

    def iterLabeledAltlexes(self, sentenceIndices=None, datumIndices=None):
        datumId = 0
        for sentenceId,labels,altlexes,pair in self.iterLabeledPairs(sentenceIndices):
            if sentenceIndices is not None and sentenceId not in sentenceIndices:
                datumId += len(altlexes)
                continue

            labelLookup = {j:i for i,j in labels}
            for i in range(len(altlexes)):
                if datumIndices is not None and datumId not in datumIndices:
                    datumId += 1
                    continue
                
                label = labelLookup.get(i, None)
                completeLabel = self.labelLookup[self.getMetaLabel(label)]

                yield sentenceId,datumId,completeLabel,altlexes[i],pair
                
                datumId += 1
            
        
    def iterAltlexes(self, sentenceIndices=None):
        for index,labels,altlexes,pair in self.iterLabeledPairs(sentenceIndices):
            lemmas = [[i.lower() for i in getLemmas(pair[0]['lemmas'])],
                      [i.lower() for i in getLemmas(pair[1]['lemmas'])]]
            pos = [getLemmas(pair[0]['pos']),
                   getLemmas(pair[1]['pos'])]

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

    def iterLabeledAltlexPairs(self, sentenceIndices=None, 
                               weighting=lambda x,y:1):
        for labels, stems in self.iterAltlexes(sentenceIndices):
            labelLookup = {j:i for i,j in labels}
            for i in range(len(stems[0])):
                label = labelLookup.get(i, None)
                metaLabel = self.getMetaLabel(label)

                stems1 = {tuple(stems[0][i])}
                stems2 = {tuple(stems[1][i])}
                
                yield metaLabel,stems1,stems2,weighting
                yield metaLabel,stems2,stems1,weighting

                datumId += 1

    def iterMetaLabels(self, sentenceIndices=None, datumIndices=None):
        for sentenceId, datumId, label, altlex, pair in self.iterLabeledAltlexes(sentenceIndices,
                                                                                 datumIndices):
            metaLabel = self.getMetaLabel(label)

            yield datumId,metaLabel
                
    def getMetaLabel(self, label):
        if label in self.reverseLabelLookup:
            return self.reverseLabelLookup[label]
        else:
            return 'other'

    def getValidSentenceIndices(self):
        #return only those indices that contain a non-empty list of labels
        for index,labels,altlexes,pair in self.iterLabeledPairs():
            if len(labels):
                yield index
            
    def getIndices(self, sentenceIndices=None, datumIndices=None, validate=False):
        indices = collections.defaultdict(list)
        for index,metaLabel in self.iterMetaLabels(sentenceIndices,
                                                   datumIndices):
            if validate and metaLabel != 'other':
                indices[metaLabel].append(index)
        return indices

    def save(self, labelsFile):
        with gzip.open(labelsFile, 'w') as f:
            json.dump([self.labels, self.altlexes, self.labelLookup], f)

    def load(self, labelsFile):
        with gzip.open(labelsFile) as f:
            self.labels, self.altlexes, self.labelLookup = json.load(f)
        self.reverseLabelLookup = {j:i for i,j in self.labelLookup.items()}
        
    def iterData(self, sentenceIndices=None, datumIndices=None, verbose=False, modBy=10000):
        for sentenceId, datumId, label, altlex, pair in self.iterLabeledAltlexes(sentenceIndices,
                                                                                 datumIndices):

            if verbose and sentenceId % modBy == 0:
                print(sentenceId)
                
            words = [[i.lower().encode('utf-8') for i in getLemmas(pair[0]['words'])],
                      [i.lower().encode('utf-8') for i in getLemmas(pair[1]['words'])]]        

            lemmas = [[i.lower().encode('utf-8') for i in getLemmas(pair[0]['lemmas'])],
                      [i.lower().encode('utf-8') for i in getLemmas(pair[1]['lemmas'])]]        
            pos = [getLemmas(pair[0]['pos']),
                   getLemmas(pair[1]['pos'])]
            ner = [getLemmas(pair[0]['ner']),
                   getLemmas(pair[1]['ner'])]
            stems = [[wordUtils.snowballStemmer.stem(wordUtils.replaceNonAscii(i)) for i in lemmas[0]],
                     [wordUtils.snowballStemmer.stem(wordUtils.replaceNonAscii(i)) for i in lemmas[1]]]

            combinedDependencies = []
            for pairIndex,half in enumerate(pair):
                partialDependencies = []
                for depIndex,dep in enumerate(half['dep']):
                    partialDependencies.append(dependencyUtils.tripleToList(dep, len(pair[pairIndex]['lemmas'][depIndex])))
                combinedDependencies.append(dependencyUtils.combineDependencies(*partialDependencies))

            #TODO: handle dependencies
            #if two sentences, don't need to really do anything (except remove altlex from deps)
            #if one sentence, call splitDependencies

            for i,which in enumerate(altlex):
                newDependencies = dependencyUtils.splitDependencies(combinedDependencies[i],
                                                                    (which[0],
                                                                     which[1]))
                dataPoint = {'altlexLength': which[1]-which[0],
                             'sentences': [{
                                 'lemmas': lemmas[i][which[0]:],
                                 'words': words[i][which[0]:],
                                 'stems': stems[i][which[0]:],
                                 'pos': pos[i][which[0]:],
                                 'ner': ner[i][which[0]:],
                                 'dependencies': newDependencies['curr']
                                 },
                                           {
                                               'lemmas': lemmas[i][:which[0]],
                                               'words': words[i][:which[0]],
                                               'stems': stems[i][:which[0]],
                                               'pos': pos[i][:which[0]],
                                               'ner': ner[i][:which[0]],
                                               'dependencies': newDependencies['prev']
                                               }],
                             'altlex': {'dependencies': newDependencies['altlex']}
                             }

                dp = DataPoint(dataPoint)
                yield sentenceId, datumId, dp, label
                    
