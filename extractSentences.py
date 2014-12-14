from __future__ import print_function

from parsedGigawordReader import ParsedGigawordReader
from sentenceReader import SentenceReader
from treeUtils import extractAltlex

import sys
import math
from collections import defaultdict
import re

if sys.version_info[0] < 3:
    class FileNotFoundError(Exception):
        pass

rwhitespace = re.compile("\s+")

def getConnectors():
    with open("/home/chidey/PDTB/Discourse/config/markers/markers") as f:
        return f.read().splitlines()

def getAltlexes():
    with open("/home/chidey/PDTB/Discourse/config/markers/altlexes") as f:
        return f.read().splitlines()

#read in causal and noncausal word pairs as dicts
#dont need numpy for this since we have sparse vectors for the sentence pairs
def getPairs(filename):
    scores = defaultdict(dict)

    with open(filename) as f:
        count = 0
        for line in f:
            count += 1
            word1,word2,score = line.split("\t")
            score = float(score)
            scores[word1][word2] = score
            if count % 100000 == 0:
                print ("done ", count)
            
    return scores

def getLength(scores):
    '''get the length of the vector'''
    total = 0
    for word1 in scores:
        for word2 in scores[word1]:
            total += scores[word1][word2]**2
            
    return math.sqrt(total)

def getCausalPairs():
    return getPairs("/home/chidey/PDTB/Discourse/working/wordpairs/gigaword2/content_word_pairs_stemmed_tsl/causal") #"/home/chidey/PDTB/Discourse/working/wordpairs/gigaword2/content_word_pairs_stemmed_tsl_tfidf/causal")

def getNonCausalPairs():
    return getPairs("/home/chidey/PDTB/Discourse/working/wordpairs/gigaword2/content_word_pairs_stemmed_tsl_tfidf/noncausal")

def getDotProduct(scores1, scores2):
    dp = 0
    for word1 in scores1:
        for word2 in scores1[word1]:
            if word1 in scores2 and word2 in scores2[word1]:
                dp += scores2[word1][word2] * scores1[word1][word2]

    return dp

#takes in a Sentence object and scores and length and returns the extraction
class CausalityScorer:
    def __init__(self):
        self.seenSentences = set()
        self.altlexes = getAltlexes()
        self.causal = getCausalPairs()
        self.causalLength = getLength(self.causal)
        '''
        nonCausal = getNonCausalPairs()
        nonCausalLength = getLength(nonCausal)
        '''

    def scoreWords(self, words1, words2):
        instance = defaultdict(dict)
        for word1 in words1:
            for word2 in words2:
                instance[word1][word2] = 1

        instanceLength = getLength(instance)
        return getDotProduct(instance, self.causal)/(self.causalLength*instanceLength)

    def scoreCausality(self, sentence, prevSentence):
        lcSentence = ' '.join(sentence.words).lower()

        if lcSentence in self.seenSentences:
            return None
        self.seenSentences.add(lcSentence)
                
        #only interested in sentences with no explicit discourse marker
        #because we already used these to create the scores
        #maybe add this later

        if prevSentence is None:
            return None

        #only interested in sentences with no altlex
        if any(lcSentence.startswith(i) for i in self.altlexes):
            return None
        
        #should also count how many of these sentences have some kind of clause as a DO
                
        #TODO nonCausalCosSim = getDotProduct(instance, nonCausal)/(nonCausalLength*instanceLength)
        #print(instanceLength, causalCosSim)

        newAltlex = extractAltlex(sentence.parse)
        if newAltlex:
            newAltlex = newAltlex.split()
            
            #otherwise make dict out of cartesian product of words
            causalCosSim = self.scoreWords(prevSentence.stems,
                                           sentence.stems[len(newAltlex):])

            '''
            instance = defaultdict(dict)
            for word1 in prevSentence.stems:
                for word2 in sentence.stems[len(newAltlex):]:
                    instance[word1][word2] = 1

            instanceLength = getLength(instance)
            causalCosSim = getDotProduct(instance, self.causal)/(self.causalLength*instanceLength)
            '''

            modAltlex = ""
            prevNamedEntity = ""
            for i in range(len(newAltlex)):
                if sentence.pos[i][0] == 'N' and sentence.ner[i] != "O":
                    if prevNamedEntity != sentence.ner[i]:
                        modAltlex += sentence.ner[i] + " "
                        prevNamedEntity = sentence.ner[i]
                elif sentence.pos[i].startswith('PR'):
                        modAltlex += sentence.pos[i] + " "
                        prevNamedEntity = ""
                elif sentence.pos[i][0].isalpha():
                        modAltlex += sentence.stems[i] + " "
                        prevNamedEntity = ""
                            
            newAltlex = ' '.join(newAltlex).lower()
            modSentence = lcSentence.replace(newAltlex, '')
            
            return ScoredSentence(causalCosSim, modAltlex, newAltlex, modSentence)
        else:
            return None



class MarkerScorer(CausalityScorer):
    def __init__(self):
        self.seenSentences = set()
        self.markers = getConnectors()
        self.markerPairs = {}
        self.markerLengths = {}
        for marker in self.markers:
            modMarker = marker.replace(' ', '_')
            try:
                self.markerPairs[marker] = getPairs('/home/chidey/PDTB/Discourse3/wordpairs/gigaword/content_word_pairs_stemmed_tfidf_adj_culled/' + modMarker)
            except IOError: #FileNotFoundError:
                continue
            except Exception:
                print('unknown exception in MarkerScorer')
                continue
            if len(self.markerPairs[marker]):
                self.markerLengths[marker] = getLength(self.markerPairs[marker])
            else:
                del self.markerPairs[marker]
                
    def scoreWords(self, words1, words2):
        instance = defaultdict(dict)
        for word1 in words1:
            for word2 in words2:
                instance[word1][word2] = 1

        instanceLength = getLength(instance)
        dotProducts = {}
        for marker in self.markerPairs:
            dotProducts[marker] = getDotProduct(instance,
                                                self.markerPairs[marker]) / \
                                                (self.markerLengths[marker] * \
                                                instanceLength)

        return dotProducts
    
class ScoredSentence:
    def __init__(self, causalCosSim, modAltlex, newAltlex, modSentence):
        self.causalCosSim = causalCosSim
        self.modAltlex = modAltlex
        self.newAltlex = newAltlex
        self.modSentence = modSentence
        
if __name__ == '__main__':
    pgr = ParsedGigawordReader(sys.argv[1])
    cs = CausalityScorer()
    
    print("causal length is ", cs.causalLength)

    altlexFile = open('newaltlexes', 'w')
    pairsFile = open('paired_bootstrapped_causal', 'w')

    try:
        for s in pgr.iterFiles():
            
            sr = SentenceReader(s)
            prevSentence = None

            for sentence in sr.iterSentences():
                ss = cs.scoreCausality(sentence, prevSentence)
                if ss is not None:
                    print("{}\t{}".format(ss.causalCosSim, ss.modAltlex), file=altlexFile)
                    print("{}\t{}\t{}\t{}".format(ss.modAltlex, ' '.join(prevSentence.words).lower(), ss.newAltlex, ss.modSentence), file=pairsFile)

                prevSentence = sentence

    finally:
        altlexFile.close()
        pairsFile.close()
