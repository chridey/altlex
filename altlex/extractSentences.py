from __future__ import print_function

from chnlp.utils.readers.parsedGigawordReader import ParsedGigawordReader
from chnlp.utils.readers.sentenceReader import SentenceReader
from chnlp.utils.treeUtils import extractAltlex
from chnlp.utils.cache import MultiWordCache

from chnlp.word2vec.model import Model

import sys
import math
from collections import defaultdict
import re
import itertools

from sklearn.externals import joblib
import numpy as np

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
            else:
                pass #print('{} and {} not found'.format(word1, word2))
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
        self.markerPairs = None
        self.markerLengths = {}
        self.loadPairs = getPairs
        self.dotProduct = getDotProduct
        self.magnitude = getLength
        
    def scoreWords(self, words1, words2, markers=None):
        if markers is None:
            markers = self.markers
            
        if self.markerPairs is None:
            self.markerPairs = {}
            
            #print("here", markers)
        for marker in markers:
            if marker not in self.markerPairs:
                modMarker = marker.replace(' ', '_')
                try:
                    self.markerPairs[marker] = getPairs('/home/chidey/PDTB/Discourse3/wordpairs/gigaword/content_word_pairs_stemmed_tfidf_adj_culled/' + modMarker)
                except IOError: #FileNotFoundError:
                    print("cant open {}".format(marker))
                    self.markerPairs[marker] = {}
                    continue
                except Exception:
                    print('unknown exception in MarkerScorer')
                    continue
                if len(self.markerPairs[marker]):
                    self.markerLengths[marker] = getLength(self.markerPairs[marker])
                #else:
                #    del self.markerPairs[marker]
        
        instance = defaultdict(dict)
        for word1 in words1:
            for word2 in words2:
                instance[word1][word2] = 1

        instanceLength = getLength(instance)
        dotProducts = {}
        for marker in markers:
            if marker in self.markerPairs and len(self.markerPairs[marker]):
            
                dotProducts[marker] = getDotProduct(instance,
                                                    self.markerPairs[marker]) / \
                                                    (self.markerLengths[marker] * \
                                                     instanceLength)

        return dotProducts

class MarkerScorerWithSimilarity(MarkerScorer):
    def __init__(self, filename):
        MarkerScorer.__init__(self)
        self.model = Model(filename)
                
    def scoreWords(self, entities1, entities2):
        #words may be words, lemmas, or stems
        #model may be words, lemmas, or stems
        #wordpairs are probably stems so we will just treat them as such for now
        #entities is a tuple of (words, lemmas, stems)

        words1,lemmas1,stems1 = entities1
        words2,lemmas2,stems2 = entities2
        instance = defaultdict(dict)
        for word1 in stems1:
            for word2 in stems2:
                instance[word1][word2] = 1
                
        dotProducts = {}
        for marker in self.markerPairs:
            dp,instance = self.dotProduct(marker, instance, entities1, entities2)

            instanceLength = getLength(instance)
            dotProducts[marker] = dp/ \
                                  (self.markerLengths[marker] * \
                                   instanceLength)

        return dotProducts

    def dotProduct(self, marker, instance, entities1, entities2):
        #if the word pair in the instance does not appear for this marker pair, use the model to predict the frequency
        words1,lemmas1,stems1 = entities1
        words2,lemmas2,stems2 = entities2

        dp = 0
        for index1,word1 in enumerate(stems1):
            if word1 not in self.markerPairs[marker]:
                neighbors1 = self.model.getNeighbors(words1[index1],lemmas1[index1],word1)
                try:
                    print("cant find {}".format(word1))
                except UnicodeEncodeError:
                    pass
            else:
                neighbors1 = [(word1, 1)]
            for index2,word2 in enumerate(stems2):
                if word2 not in self.markerPairs[marker][word1]:
                    neighbors2 = self.model.getNeighbors(words2[index2],lemmas2[index2],word2)
                    try:
                        print("cant find {}".format(word2))
                    except UnicodeEncodeError:
                        pass
                else:
                    neighbors2 = [(word2, 1)]
                #now consider the cross product of neighbors1 and neighbors2
                #take the maximum score from both

                for pair1,pair2 in sorted(itertools.product(neighbors1, neighbors2),
                                          key=lambda x:x[0][1] + x[1][1],
                                          reverse=True):
                    #print(pair1, pair2)
                    n1,s1 = pair1
                    n2,s2 = pair2
                    if n1 in self.markerPairs[marker] and n2 in self.markerPairs[marker][n1]:
                        try:
                            print("best pair is {},{} with {},{}".format(n1,n2,s1,s2))
                        except UnicodeEncodeError:
                            pass
                        #reduce by cos sim or no?
                        instance[n1][n2] = s1*s2
                        dp += self.markerPairs[marker][n1][n2]*instance[n1][n2]
                        break
        return dp,instance

class MarkerScorerWithWordPairModel(MarkerScorerWithSimilarity):
    def __init__(self, filename):
        MarkerScorerWithSimilarity.__init__(self, filename)
        
        self.model = Model(filename)
        self._wordPairModels = {}
        self._wordPairModelsCache = MultiWordCache()

    def getWordPairModelPrediction(self, marker, word1, word2):
        if not self._loadWordPairModel(marker):
            return 0

        value = self._wordPairModelsCache.lookup(marker, word1, word2)
        if value is not None:
            return value

        features = self._getFeatureVector(word1, word2)
        if features is None:
            return 0
        
        value = self._wordPairModels[marker].predict([features])[0]
        self._wordPairModelsCache.update(value, marker, word1, word2)
        return value

    def precalculateAllWordPairScores(self, sentencePairs):
        for marker in self.markers:
            self.precalculateWordPairScores(sentencePairs)
            
    def precalculateWordPairScores(self, marker, sentencePairs):
        if not self._loadWordPairModel(marker):
            return False

        wordsLookup = []
        featuresSet = []
        for sentence1,sentence2 in sentencePairs:
            for word1 in sentence1:
                for word2 in sentence2:
                    features = self._getFeatureVector(word1, word2)
                    #print(word1, word2)

                    if features is None:
                        continue
                    wordsLookup.append((word1, word2))
                    featuresSet.append(features)

        #print(len(featuresSet))
        if len(featuresSet) == 0:
            return False
        
        predictions = self._wordPairModels[marker].predict(featuresSet)
        for index,prediction in enumerate(predictions):
            word1,word2 = wordsLookup[index]
            self._wordPairModelsCache.update(prediction, marker, word1, word2)

        return True

    def clearCache(self):
        self._wordPairModelsCache.clear()

    def _getFeatureVector(self, word1, word2):
        vector1 = self.model.vector(word1)
        if vector1 is None:
            return None
        vector2 = self.model.vector(word2)
        if vector2 is None:
            return None
        return np.append(vector1, vector2)
        
    def _loadWordPairModel(self, marker):
        if marker not in self._wordPairModels:
            print(marker)
            try:
                #self._wordPairModels[marker] = joblib.load(self.model.filename + '.' + marker)
                import json
                from sklearn.linear_model import SGDRegressor
                with open(self.model.filename + '.' + marker + '.test', 'r') as f:
                    j = json.load(f)
                self._wordPairModels[marker] = SGDRegressor(**j['params'])
                self._wordPairModels[marker].coef_ = np.array(j['coef'])
                self._wordPairModels[marker].intercept_ = np.array(j['intercept'])
            except IOError:
                self._wordPairModels[marker] = None
                return False
        elif self._wordPairModels[marker] is None:
            return False
        return True
        
    def dotProduct(self, marker, instance, entities1, entities2):
        #if the word pair in the instance does not appear for this marker pair, try to find the closest word that appears for each word in the pair
        words1,lemmas1,stems1 = entities1
        words2,lemmas2,stems2 = entities2
        dp = 0

        self.precalculateWordPairScores(marker, [(stems1, stems2)])
        
        for index1,word1 in enumerate(stems1):
            for index2,word2 in enumerate(stems2):
                if word1 not in self.markerPairs[marker] or word2 not in self.markerPairs[marker][word1]:
                    freq = self.getWordPairModelPrediction(marker, word1, word2)
                else:
                    freq = self.markerPairs[marker][word1][word2]
                dp += freq

        self.clearCache()
                
        return float(dp),instance
    
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
