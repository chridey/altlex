from functools import lru_cache
from collections import defaultdict

from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet  as wn
#from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from extractSentences import CausalityScorer,MarkerScorer
from treeUtils import extractRightSiblings, extractSelfCategory, extractParentCategory
from wordNetManager import WordNetManager

def wordnet_distance(word1, word2):
    maxy = 0
    for pos in wn.VERB,wn.NOUN,wn.ADJ,wn.ADV:
        try:
            word1synset = wn.synset('{}.{}.01'.format(word1, pos))
        except WordNetError:
            continue

        try:
            word2synset = wn.synset('{}.{}.01'.format(word2, pos))
        except WordNetError:
            continue

        r = word1synset.path_similarity(word2synset)
        if r is not None and r > maxy:
            maxy = r

    return maxy

class FeatureExtractor:
    sn = SnowballStemmer('english')
    
    def __init__(self):
        with open('/home/chidey/PDTB/Discourse/config/markers/markers') as f:
            self.markers = f.read().splitlines()
        self.cs = None
        self.ms = None
        self.wn = WordNetManager()
        self.reporting = set(wn.synsets('say', pos=wn.VERB))
        self.framenetScores = {'nn' : None,
                               'vb' : None,
                               'rb' : None,
                               'jj' : None}

        self.validFeatures = {'altlex_stem' : self.getAltlexStemNgrams,
                              'curr_stem' : self.getCurrStemNgrams,
                              'prev_stem' : self.getPrevStemNgrams,
                              'reporting' : self.getReporting,
                              'final_reporting' : self.getFinalReporting,
                              'noun_similarity' : self.getNounSimilarity,
                              'coref' : self.getCoref,
                              'head_verb_altlex' : self.getHeadVerbCatAltlex,
                              'head_verb_curr' : self.getHeadVerbCatCurr,
                              'head_verb_prev' : self.getHeadVerbCatPrev,
                              'noun_cat_altlex' : self.getNounCatAltlex,
                              'noun_cat_curr' : self.getNounCatCurr,
                              'noun_cat_prev' : self.getNounCatPrev,
                              'has_copula' : self.getCopula,
                              #'pronoun' : self.getPronoun,
                              'intersection' : self.getIntersection,
                              'noun_intersection' : self.getNounIntersection,
                              'altlex_pos' : self.getAltlexPosNgrams,
                              'altlex_marker' : self.getAltlexMarker,
                              'altlex_length': self.getAltlexLength,
                              'cosine' : self.getCosineSim,
                              'marker_cosine' : self.getMarkerCosineSim,
                              'tense' : self.getTense,
                              'framenet' : self.getFramenetScore,
                              'right_siblings' : self.getRightSiblings,
                              'self_category' : self.getSelfCategory,
                              'parent_category' : self.getParentCategory,
                              'productions' : self.getProductionRules,
                              }

        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

    @property
    def experimentalSettings(self):
        return {
            #'pronoun' , #not tested
            #'tense', #doesnt help
        }

    @property
    def defaultSettings(self):
        return {
            #structural features
            'altlex_length' : True,

            #word
            'coref' : True,
            'has_copula' : True, #seems to help
            'altlex_marker' : True,
            'altlex_stem' : False, #doesnt help for NB, inconclusive for n=1 and RF
            'curr_stem' : False, #doesnt help
            'prev_stem' : False, #doesnt help

            'intersection' : False, #inconclusive
            'noun_intersection' : False, #seems to hurt

            'cosine' : False, #inconclusive
            'marker_cosine' : False, #inconclusive

            #semantic
            'final_reporting' : True,
            'reporting' : False, #inconclusive
            'head_verb_altlex' : True,
            'head_verb_curr' : True,
            'head_verb_prev' : True,
            'noun_cat_altlex' : True, #seems to help
            'noun_cat_curr' : False, #seems to hurt
            'noun_cat_prev' : False, #seems to hurt
            'noun_similarity' : False, #inconclusive
            'framenet' : True,

            #syntactic
            'altlex_pos': True,
            'right_siblings' : True,
            'self_category' : True, #seems to help
            'parent_category' : False, #doesnt help
            'productions' : False, #seems to overtrain
            }

    def getNgrams(self, featureName, gramList, n=(1,2)):
        features = {}
        prev = None
        for curr in gramList:
            if 1 in n:
                features['uni: ' + featureName + ': ' + curr] = True
            if prev:
                if 2 in n:
                    features['bi: ' + featureName + ': ' + prev + ' ' + curr] = True
            prev = curr

        return features

    @lru_cache(maxsize=None)
    def getAltlexStemNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getAltlexStemNgrams],
                              dataPoint.getAltlexStem(),
                              (1,))

    @lru_cache(maxsize=None)
    def getCurrStemNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getCurrStemNgrams],
                              dataPoint.getCurrStem())

    @lru_cache(maxsize=None)
    def getPrevStemNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getPrevStemNgrams],
                              dataPoint.getPrevStem())

    @lru_cache(maxsize=None)
    def getCoref(self, dataPoint):
        altlexLower = dataPoint.getAltlexLower()
        #also these, those
        if 'this' in altlexLower or 'that' in altlexLower \
               or 'these' in altlexLower or 'those' in altlexLower:
            coref = True
        else:
            coref = False
            
        return {self.functionFeatures[self.getCoref]:
                coref}

    @lru_cache(maxsize=None)
    def getIntersection(self, dataPoint):
        inter = len(set(dataPoint.getCurrStem()) & set(dataPoint.getPrevStem()))
        return {self.functionFeatures[self.getIntersection]:
                inter}

    @lru_cache(maxsize=None)
    def getNounIntersection(self, dataPoint):
        prevNouns = dataPoint.getStemsForPos('N', 'previous')
        altlexNouns = dataPoint.getStemsForPos('N', 'altlex')
        
        inter = len(set(prevNouns) & set(altlexNouns))
        return {self.functionFeatures[self.getNounIntersection]:
                inter}

    @lru_cache(maxsize=None)
    def getAltlexPosNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getAltlexPosNgrams],
                              dataPoint.getAltlexPos())

    @lru_cache(maxsize=None)
    def getAltlexMarker(self, dataPoint):
        features = {}
        altlexLower = dataPoint.getAltlexLower()
        altlexLowerString = ' '.join(altlexLower)
        for marker in self.markers:
            if marker in altlexLower or len(marker.split()) > 1 and marker in altlexLowerString:
                value = True
            else:
                value = False
                
            features[self.functionFeatures[self.getAltlexMarker] + ' ' + marker] = value

        return features

    def getAltlexLength(self, dataPoint):
        #batch these in groups of 5
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.altlexLength}

    @lru_cache(maxsize=None)
    def getReporting(self, dataPoint):
        #get all the verbs in the sentence and determine overlap with reporting verbs
        #or maybe just the last verb?
        
        verbs = []
        for (index,p) in enumerate(dataPoint.getAltlexPos()):
            if p[0] == 'V':
                #features['first altlex verb: ' + currStems[index]] = True
                verbs.append(dataPoint.getCurrLemmas()[index])
                break

        try:
            if len(set(wn.synsets(verbs[-1], pos=wn.VERB)) & self.reporting) > 0:
                value = True
            else:
                value = False
        except IndexError:
            #print (dataPoint.getAltlexLower(),
            #       dataPoint.getCurrParse())
            value = False

        return {self.functionFeatures[self.getReporting] :
                value}    

    @lru_cache(maxsize=None)
    def getFinalReporting(self, dataPoint):
        return {self.functionFeatures[self.getFinalReporting] :
                wordnet_distance("say",
                                 dataPoint.getAltlexLemmatized()[-1])}

    #wordnet similarity between all nouns in altlex and prev sentence
    #may provide another way of measuring coref
    @lru_cache(maxsize=None)
    def getNounSimilarity(self, dataPoint):
        prevNouns = dataPoint.getStemsForPos('N', 'previous', 'lemmas')
        altlexNouns = dataPoint.getStemsForPos('N', 'altlex', 'lemmas')

        maxy = 0
        for prev in prevNouns:
            for curr in altlexNouns:
                wnsim = self.wn.distance(prev, curr, 'N')
                if wnsim > maxy:
                    maxy = wnsim

        return {self.functionFeatures[self.getNounSimilarity] :
                maxy}
    
    @lru_cache(maxsize=None)
    def getCosineSim(self, dataPoint):
        if self.cs is None:
            self.cs = CausalityScorer()
        return {self.functionFeatures[self.getCosineSim] :
                self.cs.scoreWords(dataPoint.getPrevStem(),
                                   dataPoint.getCurrStemPostAltlex())}

    @lru_cache(maxsize=None)
    def getMarkerCosineSim(self, dataPoint):
        if self.ms is None:
            self.ms = MarkerScorer()
        scores = self.ms.scoreWords(dataPoint.getPrevStem(),
                           dataPoint.getCurrStemPostAltlex())
        ret = {}
        for marker in scores:
            ret[self.functionFeatures[self.getMarkerCosineSim] + ' ' + marker] = \
                scores[marker]

        return ret

    def _getWordNetCat(self, name, pos, lemmas):
        features = {}
        for lemma in lemmas:
            #print(pos, lemma)
            synsetCounts = defaultdict(int)
            for synset in wn.synsets(lemma,
                                     pos=self.wn.wordNetPOS[pos]):
                synsetCounts[synset] += 1
            if len(synsetCounts):
                features[name + \
                         max(synsetCounts,
                             key=synsetCounts.get).lexname()] = True

        if not len(features):
            return {name + 'None' : True}

        return features
        
    @lru_cache(maxsize=None)
    def getNounCatAltlex(self, dataPoint):
        return self._getWordNetCat(self.functionFeatures[self.getNounCatAltlex],
                                   'N', dataPoint.getStemsForPos('N',
                                                                 'altlex',
                                                                 'lemmas'))
    
    @lru_cache(maxsize=None)
    def getNounCatCurr(self, dataPoint):
        return self._getWordNetCat(self.functionFeatures[self.getNounCatCurr],
                                   'N', dataPoint.getStemsForPos('N',
                                                                 'current',
                                                                 'lemmas'))
    
    @lru_cache(maxsize=None)
    def getNounCatPrev(self, dataPoint):
        return self._getWordNetCat(self.functionFeatures[self.getNounCatPrev],
                                   'N', dataPoint.getStemsForPos('N',
                                                                 'previous',
                                                                 'lemmas'))

    #modify for nouns
    def _getHeadVerbCat(self, name, pos, lemmas):
        features = {}
        #add wordnet categories for head verb (this reduced false positives in the other test)
        #do this for 1) head of altlex and 2) head of sentence

        for (index,p) in enumerate(pos):
            if p[0] == 'V':
                #features['first altlex verb: ' + currStems[index]] = True
                #print(currLemmas[index])
                synsetCounts = defaultdict(int)
                for synset in wn.synsets(lemmas[index],
                                         pos=wn.VERB):
                    synsetCounts[synset] += 1
                if len(synsetCounts):
                    features[name + \
                             max(synsetCounts,
                                 key=synsetCounts.get).lexname()] = True

        if not len(features):
            return {name + 'None' : True}

        return features
    
    @lru_cache(maxsize=None)
    def getHeadVerbCatAltlex(self, dataPoint):
        return self._getHeadVerbCat(self.functionFeatures[self.getHeadVerbCatAltlex],
                                    dataPoint.getAltlexPos(),
                                    dataPoint.getAltlexLemmatized())

    @lru_cache(maxsize=None)
    def getHeadVerbCatCurr(self, dataPoint):
        return self._getHeadVerbCat(self.functionFeatures[self.getHeadVerbCatCurr],
                                    dataPoint.getCurrPosPostAltlex(),
                                    dataPoint.getCurrLemmasPostAltlex())
    @lru_cache(maxsize=None)
    def getHeadVerbCatPrev(self, dataPoint):
        return self._getHeadVerbCat(self.functionFeatures[self.getHeadVerbCatPrev],
                                    dataPoint.getPrevPos(),
                                    dataPoint.getPrevLemmas())

    def getCopula(self, dataPoint):
        value = False
        
        pos = dataPoint.getAltlexPos()
        words = dataPoint.getAltlex()

        for i in range(dataPoint.altlexLength):
            if pos[i][0] == 'V' and words[i] in {'is',
                                                 "'s",
                                                 'was',
                                                 'were',
                                                 'been',
                                                 'am',
                                                 'are',
                                                 'being'}:
                value = True
                break

        return {self.functionFeatures[self.getCopula]:
                value}

    def getTense(self, dataPoint):
        features = {}
        pos = dataPoint.getAltlexPos()
        words = dataPoint.getAltlex()

        for i in range(dataPoint.altlexLength):
            if pos[i][0] == 'V':
                features.update({self.functionFeatures[self.getTense] + pos[i]:
                                 True})
        return features                                 

    @lru_cache(maxsize=None)
    def getFramenetScore(self, dataPoint):
        #sum of probabilities of encoding causality for words for different parts of speech
        #stem and lowercase
        for pos in self.framenetScores:
            if self.framenetScores[pos] is None:
                self.framenetScores[pos] = {}
                with open('/home/chidey/PDTB/' + pos) as f:
                    for line in f:
                        p,word,count1,count2,score = line.split()
                        score = float(score)
                        if score > 0.0:
                            self.framenetScores[pos][self.sn.stem(word.lower())] \
                                = score

        #for now just do one aggregate score, also try different parts of speech
        #variations - only look at altlex or post altlex or entire sentence
        #           - look at previous sentence
        #           - combined score for all parts of speech
        #           - individual scores
        score = defaultdict(float)
        length = dataPoint.altlexLength
        
        for i in range(length):
            pos = dataPoint.getCurrPos()[i][:2].lower()
            stem = dataPoint.getCurrStem()[i]
            if pos in self.framenetScores and stem in self.framenetScores[pos]:
                score[''] += self.framenetScores[pos][stem]

        #print(score)
        return {self.functionFeatures[self.getFramenetScore] + pos :
                score[pos] for pos in score}

    #syntactic features based on work by Pitler, Biran
    #also consider syntactic angles
    #production rules seems to cause overtraining for such a small dataset
    #what about just production rules within the altlex?
    @lru_cache(maxsize=None)
    def getProductionRules(self, dataPoint):
        '''return the nonlexical production rules for the tree'''
        tree = dataPoint.getCurrParse()
        return {self.functionFeatures[self.getProductionRules] + str(s) :
                True for s in tree.productions() if s.is_nonlexical()}

    #consider other things to do with siblings
    #use only the immediate right sibling, unless it is a punctuation or modifying phrase
    #print all the patterns of altlexes that we see (could use for bootstrapping)
    @lru_cache(maxsize=None)
    def getRightSiblings(self, dataPoint):
        tree = dataPoint.getCurrParse()
        siblings = extractRightSiblings(dataPoint.getAltlex(), tree)

        #print(tree, siblings)
        return {self.functionFeatures[self.getRightSiblings] + s :
                True for s in siblings}

    #how about right sibling contains a VP or trace?

    @lru_cache(maxsize=None)
    def getSelfCategory(self, dataPoint):
        #could be none if altlex is not fully contained in a constituent
        cat = extractSelfCategory(dataPoint.getAltlex(),
                                  dataPoint.getCurrParse())

        return {self.functionFeatures[self.getSelfCategory] + str(cat) :
                True}
        
    @lru_cache(maxsize=None)
    def getParentCategory(self, dataPoint):
        #parent of self category
        #could be none if altlex is not fully contained in a constituent
        cat = extractParentCategory(dataPoint.getAltlex(),
                                    dataPoint.getCurrParse())

        return {self.functionFeatures[self.getParentCategory] + str(cat) :
                True}
        

    def addFeatures(self, dataPoint, featureSettings):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        features = {}
        for featureName in featureSettings:
            if featureSettings[featureName] and \
                   featureName in self.validFeatures:
                features.update(self.validFeatures[featureName](dataPoint))
        return features


#verbnet features?
#lexpar features?
