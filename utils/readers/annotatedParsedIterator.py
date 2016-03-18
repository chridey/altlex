import os
import gzip
import json

from altlex.utils import wordUtils
from altlex.utils.readers.parseMetadata import ParseMetadata

class AnnotatedParsedIterator:
    def __init__(self, indir, markedData, altlexes, labelLookup, verbose=False, wordsOnly=False):
        self.indir = indir
        self.markedData = markedData
        self.altlexes = altlexes
        self.labelLookup = labelLookup
        self.verbose = verbose
        self.wordsOnly = wordsOnly
        self._unfoundAltlexes = 0
        self._totalAltlexes = 0
        
    def __iter__(self):
        for filename in sorted(os.listdir(self.indir)):
            if self.verbose:
                print(filename)

            if not filename.endswith('.gz'):
                continue

            with gzip.open(os.path.join(self.indir, filename)) as f:
                j = json.load(f)

            articleIndex,wikiIndex,junk = filename.split('.', 2)

            for sentenceIndex,sentence in enumerate(j):
                yield int(articleIndex),int(wikiIndex),sentenceIndex,sentence

    def iterData(self, sentenceIndices=None, datumIndices=None, modBy=False, textOnly=False):
        datumIndex = 0
        self._unfoundAltlexes = 0
        self._totalAltlexes = 0
        
        for articleIndex,wikiIndex,sentenceIndex,sentence in self.__iter__():
            parse = ParseMetadata(sentence)

            #TODO: get altlex
            #wordUtils.findPhrase
            goldAltlexes = {}
            if (articleIndex,wikiIndex) in self.markedData:
                altlexes = self.markedData[(articleIndex,wikiIndex)][sentenceIndex][0]
                
                for phrase,metaLabel in altlexes:
                    start = wordUtils.findPhrase(phrase, parse.words)
                    if start is None:
                        print('Problem with finding phrase {} in lemmas {}'.format(phrase, parse.words))
                    else:
                        for i in range(start, start+len(phrase)):
                            goldAltlexes[i] = (start, start+len(phrase), metaLabel)

            #go through and find matches for any known altlexes
            words = parse.wordsLower
            lemmas = parse.lemmasLower
            pos = parse.pos
            knownConnectives = []

            if self.altlexes is not None:
                foundIndices = set()
                #go from the longest length to the shortest
                for length in range(len(lemmas))[::-1]:
                    for i in range(len(lemmas)-length):
                        j = i+length

                        if self.wordsOnly:
                            l = tuple(words[i:j])
                        else:
                            l = tuple(lemmas[i:j] + pos[i:j])
                            
                        if i not in foundIndices and j not in foundIndices:
                            if l in self.altlexes:
                                knownConnectives.append((i,j))
                                foundIndices.update(set(range(i,j)))

            foundAltlexes = set()
            for connective in knownConnectives:
                for i in range(connective[0], connective[1]):
                    if i in goldAltlexes:
                        metaLabel = goldAltlexes[i][-1]
                        foundAltlexes.add(goldAltlexes[i])
                        self._totalAltlexes += 1
                        break
                    else:
                        metaLabel = 'notcausal'

                datumIndex += 1
                if textOnly:
                    prevWords = ' '.join(parse.words[:connective[0]])
                    altlex = ' '.join(parse.words[connective[0]:connective[1]])
                    currWords = ' '.join(parse.words[connective[1]:])
                    
                    yield sentenceIndex, datumIndex, prevWords, altlex, currWords
                else:
                    try:
                        yield sentenceIndex, datumIndex, parse.datapoint(*connective), self.labelLookup[metaLabel]
                    except KeyError:
                        print('Problem with {} {} metalabel "{}"'.format(sentenceIndex, datumIndex, metaLabel))

            #handle the altlexes that are not found
            #TODO: add to self._unfoundAltlexes
            for altlex in set(goldAltlexes.values()) - foundAltlexes:
                self._unfoundAltlexes += 1
                if self.verbose:
                    print('altlex {} not found in {}'.format(altlex,
                                                             sentence['words']))
    @property
    def unfoundAltlexes(self):
        #if no defined, call iterData
        
        return self._unfoundAltlexes

    @property
    def totalAltlexes(self):
        #if no defined, call iterData
        
        return self._totalAltlexes
