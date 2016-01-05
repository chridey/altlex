import os
import math
import json
import sys
import itertools

if sys.version_info > (3,):
    from functools import lru_cache
else:
    from functools32 import lru_cache

from collections import defaultdict

import numpy
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib

from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet  as wn
#from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import gensim
#from chnlp.nn.causalRNN import model
#from chnlp.nn.discourseRNN import model as discourseModel

from chnlp.altlex.dataPoint import DataPoint
from chnlp.altlex.extractSentences import CausalityScorer,MarkerScorer,MarkerScorerWithSimilarity,MarkerScorerWithWordPairModel
from chnlp.utils.treeUtils import extractRightSiblings, extractSelfParse, extractSelfCategory, extractParentCategory
from chnlp.utils.utils import makeNgrams
from chnlp.semantics.wordNetManager import WordNetManager
from chnlp.semantics.verbNetManager import VerbNetManager
from chnlp.semantics.frameNetManager import FrameNetManager

from chnlp.altlex.taggedSet import TaggedDataPoint

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

def makeDataset(data, featureExtractor, featureSettings, max=float('inf'), preprocessed=False,
                invalidLabels = {}):
    
    taggedSet = []

    #for some features it is faster to preprocess the entire dataset
    if featureSettings.get('kld_latent_factors', False):
        featureExtractor.preprocessKldFeatures(data)
            
    for i,dataPoint in enumerate(data):
        if i >= max:
            break
        #print(dataPoint)
        
        #add pair of features dictionary and True or False
        if preprocessed:
            if dataPoint[1] in invalidLabels:
                continue #label = 0
            else:
                label = dataPoint[1]
            features = {i:j for i,j in dataPoint[0].items() if any(i.startswith(f) for f in featureSettings)}
            taggedDataPoint = TaggedDataPoint((features, label))
        else:
            dp = DataPoint(dataPoint)
    
            features = featureExtractor.addFeatures(dp, featureSettings)

            if dp.getTag() == 'causal':
                taggedDataPoint = TaggedDataPoint((features, True))
            elif dp.getTag() is None:
                taggedDataPoint = TaggedDataPoint((features, -1))
            else:
                taggedDataPoint = TaggedDataPoint((features, False))

            taggedDataPoint.addData(dp.data)

        taggedSet.append(taggedDataPoint)
        
    return taggedSet

cacheSize = 1000
class FeatureExtractor:
    sn = SnowballStemmer('english')
    configPath = os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'config')
    gensimFile = '/local/nlp/chidey/model.1428650472.31.word2vec.curr.nelemmas'
    theanoDir = '/local/nlp/chidey/causal_rnn_2layer/'
    theanoDirDiscourse = '/local/nlp/chidey/discourse_rnn_2layer/'
    
    def __init__(self):
        self.pos = None
        
        with open(os.path.join(FeatureExtractor.configPath, 'markers')) as f:
            self.markers = f.read().splitlines()
        self.cs = None
        self.ms = None
        self.msws = None
        self.mswwpm = None
        
        self.wn = WordNetManager()
        self.vn = VerbNetManager()

        self.reporting = set(wn.synsets('say', pos=wn.VERB))

        self.hedgingScores = None

        self.framenetScores = {'nn' : None,
                               'vb' : None,
                               'rb' : None,
                               'jj' : None,
                               'nn_anticausal' : None,
                               'vb_anticausal' : None,
                               'rb_anticausal' : None,
                               'jj_anticausal' : None}
        self.framenetFrames = None
        
        self.pathSimilaritySeeds = {'comparison': {'compare', 'contrast', 'oppose', 'concede'}, 'expansion': {'restate',
                                    'expand', 'specify', 'expect', 'instantiate', 'list'}}
        
        self.wtmfLookup = None
        self.wtmfWordMatrix = None
        self.wtmfSeedMatrix = None
        self.svdLookup = None
        self.svdLearner = None
        self.svdCausal = None

        self.verbClusters = None

        self.gensimModel = None
        self.theanoModel = None
        self.theanoModelDiscourse = None
        
        self.validFeatures = {'altlex_stem' : self.getAltlexStemNgrams,
                              'curr_stem' : self.getCurrStemNgrams,
                              'prev_stem' : self.getPrevStemNgrams,
                              'unistems' : self.getUnistems,
                              'bistems' : self.getBistems,
                              'unistems_sep' : self.getUnistemsSep,
                              'bistems_sep' : self.getBistemsSep,
                              'stem_pairs' : self.getStemPairs,
                              'all_stem_pairs' : self.getAllStemPairs,
                              'connective_pairs' : self.getConnectivePairs,
                              'reporting' : self.getReporting,
                              'final_reporting' : self.getFinalReporting,
                              'final_time' : self.getFinalTime,
                              'final_example' : self.getFinalExample,
                              'noun_similarity' : self.getNounSimilarity,
                              'coref' : self.getCoref,
                              'head_verb_altlex' : self.getHeadVerbCatAltlex,
                              'head_verb_curr' : self.getHeadVerbCatCurr,
                              'head_verb_prev' : self.getHeadVerbCatPrev,
                              'noun_cat_altlex' : self.getNounCatAltlex,
                              'noun_cat_curr' : self.getNounCatCurr,
                              'noun_cat_prev' : self.getNounCatPrev,
                              'verbnet_class_altlex' : self.getVerbNetClassesAltlex,
                              'verbnet_class_curr' : self.getVerbNetClassesCurrent,
                              'verbnet_class_prev' : self.getVerbNetClassesPrevious,
                              'theme_role_altlex' : self.getThematicRolesAltlex,
                              'theme_role_curr' : self.getThematicRolesCurrent,
                              'theme_role_prev' : self.getThematicRolesPrevious,
                              'has_copula' : self.getCopula,
                              #'pronoun' : self.getPronoun,
                              'intersection' : self.getIntersection,
                              'noun_intersection' : self.getNounIntersection,
                              'altlex_pos' : self.getAltlexPosNgrams,
                              'first_pos' : self.getFirstPos,
                              'altlex_marker' : self.getAltlexMarker,
                              'altlex_length': self.getAltlexLength,
                              'curr_length': self.getCurrLength,
                              'prev_length': self.getPrevLength,
                              'curr_length_post_altlex': self.getCurrPostLength,
                              'cosine' : self.getCosineSim,
                              'marker_cosine' : self.getMarkerCosineSim,
                              'marker_cosine_with_similarity' : self.getMarkerCosineSimWithSim,
                              'marker_cosine_with_wpmodel' : self.getMarkerCosineSimWithWordPairModel,
                              'tense' : self.getTense,
                              'framenet' : self.getFramenetScore,
                              'framenet_frames' : self.getFramenetFrames,
                              'right_siblings' : self.getRightSiblings,
                              'self_category' : self.getSelfCategory,
                              'parent_category' : self.getParentCategory,
                              'productions' : self.getProductionRules,
                              'altlex_productions' : self.getAltlexProductionRules,
                              'altlex_nouns' : self.getAltlexNouns,
                              'altlex_verbs' : self.getAltlexVerbs,
                              'hedging' : self.getHedgingScore,
                              'wtmf': self.getWTMFScore,
                              'svd': self.getSVDScore,
                              'full_altlex': self.getFullAltlex,
                              'path_sim': self.getPathSimilarity,
                              'verb_cluster': self.getVerbCluster,
                              'coherence': self.getCoherence,
                              'neurocausal': self.getNeuroCausal,
                              'neurodiscourse': self.getNeuroDiscourse,
                              'head_verb_pair': self.getHeadVerbPair,
                              'head_verb_cat_pair': self.getHeadVerbCatPair,
                              'all_verb_cat_pair': self.getAllVerbCatPair,                      
                              'head_verb_net_pair': self.getHeadVerbNetPair,
                              'ordered_data': self.orderedData,
                              'kld_latent_factors': self.kldLatentFactors,
                              }

        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

    @property
    def structuralFeatures(self):
        return {
            'altlex_length' : True,
            'curr_length' : False,
            'prev_length' : False,
            'curr_length_post_altlex' : False
        }

    @property
    def lexicalFeatures(self):
        return {
            'coref' : True,
            'has_copula' : True, #seems to help
            'altlex_marker' : True,
            'altlex_stem' : True, #doesnt help for NB, inconclusive for n=1 and RF
            'full_altlex' : False,
            'curr_stem' : False, #doesnt help
            'prev_stem' : False, #doesnt help

            'altlex_nouns' : False,
            'altlex_verbs' : False,

            'intersection' : False, #inconclusive
            'noun_intersection' : False, #seems to hurt

            'cosine' : False, #inconclusive
            'marker_cosine' : False, #inconclusive

            'wtmf': False,
            'svd': False,
        }

    @property
    def syntacticFeatures(self):
        return {
            'altlex_pos': False,#True,
            'first_pos' : False,
            'right_siblings' : False,#True,
            'self_category' : False,#True, #seems to help
            'parent_category' : False, #doesnt help
            'productions' : False, #seems to overtrain
            'altlex_productions' : False, #inconclusive, but seems worse
        }

    @property
    def semanticFeatures(self):
        return {
            'final_reporting' : True,
            'final_time' : True,
            'final_example' : False,
            'reporting' : False, #inconclusive, mod final reporting may capture
            'head_verb_altlex' : True,
            'head_verb_curr' : True,
            'head_verb_prev' : True,
            'noun_cat_altlex' : False,#True, #seems to help
            'noun_cat_curr' : False, #seems to hurt
            'noun_cat_prev' : False, #seems to hurt
            'noun_similarity' : False, #inconclusive
            'verbnet_class_prev' : True,
            'verbnet_class_curr' : True, #both seem to help
            'verbnet_class_altlex' : False,
            'theme_role_altlex' : False,
            'theme_role_curr' : False,
            'theme_role_prev' : False, #helps for LR but not RF but why??
            'framenet' : True,
            'framenet_frames' : False,
            'hedging' : False,
            'path_sim': False,#True,
            'neurocausal': False,
        }

    @property
    def coherenceFeatures(self):
        return {
            'coherence': True,
            }
    
    @property
    def defaultSettings(self):
        #structural features
        ret = {s:False for s in self.validFeatures}
        
        ret.update(self.structuralFeatures)

        #word
        ret.update(self.lexicalFeatures)
        
        #syntactic
        ret.update(self.syntacticFeatures)
        
        #semantic
        ret.update(self.semanticFeatures)
        
        return ret

    @property
    def taggingSettings(self):
        ret = self.lexicalFeatures
        ret.update(self.syntacticFeatures)
        ret.update({'final_time': True,
                    'final_example': True,
                    'head_verb_altlex': True,
                    'verbnet_class_prev': True,
                    'verbnet_class_curr': True,
                    'verbnet_class_altlex': True,
                    'framenet': True,
                    #added syntactic features
                    'parent_category': True})
        
        return ret

    def featureSubsets(self, featureSubset):
        if featureSubset == 'structural':
            return self.structuralFeatures
        elif featureSubset == 'lexical':
            return self.lexicalFeatures
        elif featureSubset == 'syntactic':
            return self.syntacticFeatures
        elif featureSubset == 'semantic':
            return self.semanticFeatures
        else:
            raise NotImplementedError

    def getFullAltlex(self, dataPoint):
        return {self.functionFeatures[self.getFullAltlex] + '_' + ' '.join(dataPoint.getAltlexLower()):True}
        
    def getNgrams(self, featureName, gramList, n=(1,)):
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

    @lru_cache(maxsize=cacheSize)
    def getAltlexStemNgrams(self, dataPoint):
        #CHANGE
        #hasNgram = False
        #if 'other' in dataPoint.getAltlexStem() or 'anoth' in dataPoint.getAltlexStem():
        #    hasNgram = True
        #    
        #return {self.functionFeatures[self.getAltlexStemNgrams]: hasNgram}
        
        return self.getNgrams(self.functionFeatures[self.getAltlexStemNgrams],
                              dataPoint.getAltlexStem(),
                              (1,2)) #CHANGE

    @lru_cache(maxsize=cacheSize)
    def getCurrStemNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getCurrStemNgrams],
                              dataPoint.getCurrStem())

    @lru_cache(maxsize=cacheSize)
    def getPrevStemNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getPrevStemNgrams],
                              dataPoint.getPrevStem())

    def orderedData(self, dataPoint):
        lemmas1 = dataPoint.getPrevLemmas()
        lemmas2 = dataPoint.getCurrLemmas()
        return {'prev_lemmas':lemmas1,
                'curr_lemmas':lemmas2}

    def getNstems(self, dataPoint, n, prefix=False):
        if prefix:
            first = '1_'
            second = '2_'
        else:
            first = ''
            second = ''
        prev = list(makeNgrams([first+i for i in dataPoint.getPrevStem()], n))
        curr = list(makeNgrams([second+i for i in dataPoint.getCurrStem()], n))        
        return dict(zip(prev+curr, [True]*(len(prev)+len(curr))))
    
    def getUnistems(self, dataPoint):
        return self.getNstems(dataPoint, 1)

    def getBistems(self, dataPoint):
        return self.getNstems(dataPoint, 2)

    def getUnistemsSep(self, dataPoint):
        return self.getNstems(dataPoint, 1, True)

    def getBistemsSep(self, dataPoint):
        return self.getNstems(dataPoint, 2, True)

    def getStemPairs(self, dataPoint):
        features = {}
        '''
        pos1 = dataPoint.getPrevPos()
        pos2 = dataPoint.getCurrPos()
        pos = {chr(i) for i in range(0,128)}
        #pos = {chr(i) for i in range(65,91)} | {'.', ','}
        #pos = {'V', 'N', 'I'}
        if self.pos is None:
            self.pos = set()
        
        #try just first words
        features= {'first: {}_{}'.format(dataPoint.getPrevStem()[0],
                                      dataPoint.getCurrStem()[0]): True}
        #features.update( {'last: {}_{}'.format(dataPoint.getPrevStem()[-1],
        #                             dataPoint.getCurrStem()[-1]): True})
        if len(dataPoint.getPrevStem()) > 1:
            features.update( {'continue: {}_{}'.format(dataPoint.getPrevStem()[-2],
                                                       dataPoint.getCurrStem()[0]): True} )
        features.update( {'first word: {}'.format(dataPoint.getCurrStem()[0]): True} )
        #features.update( {'last word: {}'.format(dataPoint.getPrevStem()[-1]): True} )
        #for stem1 in dataPoint.getPrevStem()[:-1]:
        #    features['prev stem: {}'.format(stem1)] = True
        #for stem2 in dataPoint.getCurrStem()[1:]:
        #    features['curr stem: {}'.format(stem2)] = True
            
        return features
        '''
        #only include those with connective
        pairs = list(itertools.product(['1_'+i for i in dataPoint.getPrevStem()],
                                       ['2_'+i for i in dataPoint.getCurrStem()]))
        return dict(zip(pairs, [True]*len(pairs)))

    def getAllStemPairs(self, dataPoint):
        pairs = list(itertools.permutations(dataPoint.getPrevStem()+dataPoint.getCurrStem(), 2))
        return dict(zip(pairs, [True]*len(pairs)))
    
    def getConnectivePairs(self, dataPoint):
        curr = [i.lower() for i in dataPoint.getCurrLemmas()]
        prev = [i.lower() for i in dataPoint.getPrevLemmas()]
        currMarkers = []
        prevMarkers = []
        for marker in self.markers:
            if marker in curr or len(marker.split()) > 1 and marker in ' '.join(curr):
                currMarkers.append(marker)
            if marker in prev or len(marker.split()) > 1 and marker in ' '.join(prev):
                prevMarkers.append(marker)

        features = {}
        for m1 in prevMarkers:
            for m2 in currMarkers:
                features['conn: {}_{}'.format(m1,m2)] = True
        return features
                
    @lru_cache(maxsize=cacheSize)
    def getAltlexNouns(self, dataPoint):
        return {self.functionFeatures[self.getAltlexNouns] + noun: True \
                for noun in dataPoint.getStemsForPos('N', 'altlex')}

    @lru_cache(maxsize=cacheSize)
    def getAltlexVerbs(self, dataPoint):
        return {self.functionFeatures[self.getAltlexVerbs] + verb: True \
                for verb in dataPoint.getStemsForPos('V', 'altlex')}

    @lru_cache(maxsize=cacheSize)
    def getCoref(self, dataPoint):
        altlex = dataPoint.getAltlex()
        altlexPos = dataPoint.getAltlexPos()

        coref = False
        for index,word in enumerate(altlex):
            #only count when these are determiners
            if word.lower() in {'this', 'that', 'these', 'those'} and \
               altlexPos[index] == 'DT':
                coref = True
                break
                
        '''
        if 'this' in altlexLower or 'that' in altlexLower \
               or 'these' in altlexLower or 'those' in altlexLower:
            coref = True
        else:
            coref = False
        '''
        
        return {self.functionFeatures[self.getCoref]:
                coref}

    @lru_cache(maxsize=cacheSize)
    def getIntersection(self, dataPoint):
        inter = len(set(dataPoint.getCurrStem()) & set(dataPoint.getPrevStem()))
        return {self.functionFeatures[self.getIntersection]:
                inter}

    @lru_cache(maxsize=cacheSize)
    def getNounIntersection(self, dataPoint):
        prevNouns = dataPoint.getStemsForPos('N', 'previous')
        altlexNouns = dataPoint.getStemsForPos('N', 'altlex')
        
        inter = len(set(prevNouns) & set(altlexNouns))
        return {self.functionFeatures[self.getNounIntersection]:
                inter}

    @lru_cache(maxsize=cacheSize)
    def getAltlexPosNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getAltlexPosNgrams],
                              dataPoint.getAltlexPos())

    @lru_cache(maxsize=cacheSize)
    def getFirstPos(self, dataPoint):
        return {self.functionFeatures[self.getFirstPos] + dataPoint.getAltlexPos()[0] : True}

    @lru_cache(maxsize=cacheSize)
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

    def getCurrLength(self, dataPoint):
        #batch these in groups of 5
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.currSentenceLength}

    def getPrevLength(self, dataPoint):
        #batch these in groups of 5
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.prevSentenceLength}

    def getCurrPostLength(self, dataPoint):
        #batch these in groups of 5
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.currSentenceLengthPostAltlex}

    @lru_cache(maxsize=cacheSize)
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

    @lru_cache(maxsize=cacheSize)
    def getFinalReporting(self, dataPoint):
        altlex = dataPoint.getAltlexLemmatized()
        if len(altlex):
            maxDist = max(wordnet_distance("say",
                                           lemma)
                          for lemma in dataPoint.getAltlexLemmatized())
        else:
            maxDist = 0

        return {self.functionFeatures[self.getFinalReporting] :
                maxDist}


    @lru_cache(maxsize=cacheSize)
    def getPathSimilarity(self, dataPoint):
        altlex = dataPoint.getAltlexLemmatized()
        features = {}
        if len(altlex):
            #print(altlex)
            for discourseClass in self.pathSimilaritySeeds:
                x = 0
                for seed in self.pathSimilaritySeeds[discourseClass]:
                    maxDist = max(wordnet_distance(seed,
                                                   lemma)
                                  for lemma in dataPoint.getAltlexLemmatized())
                    if maxDist > x:
                        x = maxDist
                features[self.functionFeatures[self.getFinalReporting] + '_' + discourseClass] = x
                #print (discourseClass, x)
        return features

    @lru_cache(maxsize=cacheSize)
    def getFinalTime(self, dataPoint):
        altlex = dataPoint.getAltlexLemmatized()
        if len(altlex):
            distance = wordnet_distance("time",
                                        altlex[-1])
        else:
            distance = 0
        return {self.functionFeatures[self.getFinalTime] :
                distance}

    @lru_cache(maxsize=cacheSize)
    def getFinalExample(self, dataPoint):
        altlex = dataPoint.getAltlexLemmatized()
        if len(altlex):
            distance = wordnet_distance("example",
                                        altlex[-1])
        else:
            distance = 0
        return {self.functionFeatures[self.getFinalExample] :
                distance}

    #wordnet similarity between all nouns in altlex and prev sentence
    #may provide another way of measuring coref
    @lru_cache(maxsize=cacheSize)
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
    
    @lru_cache(maxsize=cacheSize)
    def getCosineSim(self, dataPoint):
        if self.cs is None:
            self.cs = CausalityScorer()

        score = self.cs.scoreWords(dataPoint.getPrevStem(),
                                   dataPoint.getCurrStemPostAltlex())

        print(dataPoint.getPrevStem(),
              dataPoint.getCurrStemPostAltlex(),
              score)
        
        return {self.functionFeatures[self.getCosineSim] : score}
    
    @lru_cache(maxsize=cacheSize)
    def getMarkerCosineSim(self, dataPoint):
        if self.ms is None:
            self.ms = MarkerScorer()
        #if dataPoint.altlexLength > 0:
        #    print(dataPoint.getPrevStem(),
        #          dataPoint.getCurrStemPostAltlex())
        scores = self.ms.scoreWords(dataPoint.getPrevStem(),
                                    dataPoint.getCurrStemPostAltlex())
        ret = {}
        for marker in scores:
            ret[self.functionFeatures[self.getMarkerCosineSim] + ' ' + marker] = \
                scores[marker]

        return ret

    @lru_cache(maxsize=cacheSize)
    def getMarkerCosineSimWithSim(self, dataPoint):
        if self.msws is None:
            self.msws = MarkerScorerWithSimilarity('/local/nlp/chidey/model.1427199755.85.word2vec.curr.lemmas')
        entity1 = dataPoint.getPrevWords(),dataPoint.getPrevLemmas(),dataPoint.getPrevStem()
        entity2 = dataPoint.getCurrWordsPostAltlex(),dataPoint.getCurrLemmasPostAltlex(),dataPoint.getCurrStemPostAltlex()
    
        scores = self.msws.scoreWords(entity1, entity2)
        
        ret = {}
        for marker in scores:
            ret[self.functionFeatures[self.getMarkerCosineSimWithSim] + ' ' + marker] = \
                scores[marker]

        return ret
    
    @lru_cache(maxsize=cacheSize)
    def getMarkerCosineSimWithWordPairModel(self, dataPoint):
        if self.mswwpm is None:
            self.mswwpm = MarkerScorerWithWordPairModel('/local/nlp/chidey/model.1427285719.53.word2vec.curr.stems')
        entity1 = dataPoint.getPrevWords(),dataPoint.getPrevLemmas(),dataPoint.getPrevStem()
        entity2 = dataPoint.getCurrWordsPostAltlex(),dataPoint.getCurrLemmasPostAltlex(),dataPoint.getCurrStemPostAltlex()
    
        scores = self.mswwpm.scoreWords(entity1, entity2)
        
        ret = {}
        for marker in scores:
            ret[self.functionFeatures[self.getMarkerCosineSimWithWordPairModel] + ' ' + marker] = \
                scores[marker]

        return ret

    def _getWordNet(self, name, pos, lemmas):
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
        
    @lru_cache(maxsize=cacheSize)
    def getNounCatAltlex(self, dataPoint):
        return self._getWordNet(self.functionFeatures[self.getNounCatAltlex],
                                   'N', dataPoint.getStemsForPos('N',
                                                                 'altlex',
                                                                 'lemmas'))
    
    @lru_cache(maxsize=cacheSize)
    def getNounCatCurr(self, dataPoint):
        return self._getWordNet(self.functionFeatures[self.getNounCatCurr],
                                   'N', dataPoint.getStemsForPos('N',
                                                                 'current',
                                                                 'lemmas'))
    
    @lru_cache(maxsize=cacheSize)
    def getNounCatPrev(self, dataPoint):
        return self._getWordNet(self.functionFeatures[self.getNounCatPrev],
                                   'N', dataPoint.getStemsForPos('N',
                                                                 'previous',
                                                                 'lemmas'))

    #modify for nouns
    def _getWordNetCat(self, pos, lemmas, search):
        cats = []
        for (index,p) in enumerate(pos):
            if p[0] == search:
                #features['first altlex verb: ' + currStems[index]] = True
                #print(currLemmas[index])
                synsetCounts = defaultdict(int)
                for synset in wn.synsets(lemmas[index],
                                         pos=wn.VERB):
                    synsetCounts[synset] += 1
                if len(synsetCounts):
                    lexCat = max(synsetCounts,
                                 key=synsetCounts.get).lexname()
                    #CHANGE
                    #if 'verb.' + lexCat in {'stative', 'communication', 'change', 'contact', 'cognition', 'possession', 'social', 'motion'}:
                    #features[name + \
                    #         lexCat] = True
                    cats.append(lexCat)

        return cats
    
    def _getHeadVerbCat(self, name, pos, lemmas):
        features = {}
        #add wordnet categories for head verb (this reduced false positives in the other test)
        #do this for 1) head of altlex and 2) head of sentence
        cats = self._getWordNetCat(pos, lemmas, 'V')
        
        if not len(cats):
            return {name + 'None' : True}

        return {name + cat: True for cat in cats}
    
    @lru_cache(maxsize=cacheSize)
    def getHeadVerbCatAltlex(self, dataPoint):
        return self._getHeadVerbCat(self.functionFeatures[self.getHeadVerbCatAltlex],
                                    dataPoint.getAltlexPos(),
                                    dataPoint.getAltlexLemmatized())

    @lru_cache(maxsize=cacheSize)
    def getHeadVerbCatCurr(self, dataPoint):
        return self._getHeadVerbCat(self.functionFeatures[self.getHeadVerbCatCurr],
                                    dataPoint.getCurrPosPostAltlex(),
                                    dataPoint.getCurrLemmasPostAltlex())
    @lru_cache(maxsize=cacheSize)
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

    def _loadWTMF(self):
        with open(os.path.join(FeatureExtractor.configPath,
                               'wtmf_lookup.json')) as f:
            lookup = json.load(f)
        self.wtmfLookup = lookup['words']

        with open(os.path.join(FeatureExtractor.configPath,
                               'wtmf_matrix.json')) as f:
            latentMatrices = json.load(f)
            
        self.wtmfWordMatrix, self.wtmfSeedMatrix = latentMatrices
        
    def getWTMFScore(self, dataPoint):
        altlex = dataPoint.getAltlexStem()
        curr = dataPoint.getCurrStemPostAltlex()

        if self.wtmfLookup is None:
            self._loadWTMF()

        features = {}            
        for name,stems in (('_altlex', altlex),
                           #('_curr', curr)
                           ):

            latentVector = [0 for i in range(len(self.wtmfWordMatrix[0]))]
            for stem in stems:
                if stem in self.wtmfLookup:
                    arrayIndex = self.wtmfLookup[stem]
                    stemLatentVector = self.wtmfWordMatrix[arrayIndex]
                    #print(stem, stemLatentVector)
                    latentVector = numpy.add(latentVector, stemLatentVector)
            #get vector length
            #print(latentVector)
            numpy.divide(latentVector, len(stems))
            causalVector = numpy.dot(self.wtmfSeedMatrix, numpy.transpose([latentVector]))
            print(stems, causalVector)
            score = math.sqrt(numpy.dot(numpy.transpose(causalVector), causalVector))
            features.update({self.functionFeatures[self.getWTMFScore] + name:
                             score})

        return features

    @lru_cache(maxsize=cacheSize)
    def getSVDScore(self, dataPoint):
        altlex = dataPoint.getAltlexStem()
        curr = dataPoint.getCurrStemPostAltlex()

        if self.svdLookup is None:
            with open(os.path.join(FeatureExtractor.configPath,
                                   'causal_cooccurrence_lookup.json')) as f:
                lookup = json.load(f)
            self.svdLookup = lookup['words']

        if self.svdLearner is None:
            self.svdLearner = joblib.load(os.path.join(FeatureExtractor.configPath,
                                                       'causal_cooccurrence_svd'))
            print(self.svdLearner.components_.shape)

        if self.svdCausal is None:
            self.svdCausal = numpy.load(os.path.join(FeatureExtractor.configPath,
                                                     'causal_cooccurrence_svd.npy'))
            print(self.svdCausal.shape)
            
        scores = {}
        for name,stems in (('_altlex', altlex),
                           #('_curr', curr)
                           ):
            latentVector = [0 for i in range(len(self.svdLookup))]
            for stem in stems:
                if stem in self.svdLookup:
                    arrayIndex = self.svdLookup[stem]
                    latentVector[arrayIndex] = 1
            latentVector = self.svdLearner.transform([latentVector])
            #print(latentVector.shape)
            scores[name] = numpy.amax(cosine_similarity(latentVector, numpy.transpose(self.svdCausal)))

        return {self.functionFeatures[self.getSVDScore]:
                scores[name] for name in scores}

        #Utest = np.dot(Xtest * V.T, inv(np.diag(Sigma)))

    @lru_cache(maxsize=cacheSize)
    def getHedgingScore(self, dataPoint):
        #sum of probabilities of encoding belief, might be related to causality
        #stem and lowercase
        if self.hedgingScores is None:
            self.hedgingScores = defaultdict(float)
            with open(os.path.join(FeatureExtractor.configPath,
                                   'hedging')) as f:
                for line in f:
                    word,count = line.split()
                    self.hedgingScores[self.sn.stem(word.lower())] += float(count)

            sumCount = sum(self.hedgingScores.values())
            for word in self.hedgingScores:
                self.hedgingScores[word] /= sumCount

        #for now just do one aggregate score
        score = 0.0
        stems = dataPoint.getCurrStem()
        for stem in stems:
            if stem in self.hedgingScores:
                score += self.hedgingScores[stem]

        return {self.functionFeatures[self.getHedgingScore]:
                bool(score)}

    def _getCausalFrames(self, frames, name):
        features = {}

        hasCausal = False
        hasAntiCausal = False

        for role in frames:
            if self.framenetFrames.isCausalFrame(role):
                hasCausal = True
                break
            if self.framenetFrames.isAntiCausalFrame(role):
                hasAntiCausal = True
                break
        features[name + '_has_causal'] = hasCausal
        features[self.functionFeatures[self.getFramenetScore] + '_has_anticausal'] = hasAntiCausal

        return features

    @lru_cache(maxsize=cacheSize)
    def getFramenetFrames(self, dataPoint):
        #TODO: check range of arguments
        #DONEish: create hierarchy of frames
        #TODO: create production rules
        
        if self.framenetFrames is None:
            self.framenetFrames = FrameNetManager(os.path.join(os.path.join(FeatureExtractor.configPath,
                                                                            'framenet'), 'frame_parsed_data'))

        features = {}

        text1 = ''.join(dataPoint.getPrevWords())
        try:
            frames1 = self.framenetFrames.getFrames(text1)
                
        except KeyError:
            print("Can't find " + text1)
            frames1 = {}

        #features.update(self._getCausalFrames(frames1, self.functionFeatures[self.getFramenetScore] + '_prev'))
            
        features.update({self.functionFeatures[self.getFramenetScore] + '_prev_' + frame : True for frame in frames1})

        text2 = ''.join(dataPoint.getCurrWords())
        try:
            frames2 = self.framenetFrames.getFrames(text2)
        except KeyError:
            print("Can't find " + text2)
            frames2 = {}

        #features.update(self._getCausalFrames(frames2, self.functionFeatures[self.getFramenetScore] + '_curr'))
            
        features.update({self.functionFeatures[self.getFramenetScore] + '_curr_' + frame : True for frame in frames2})
        
        return features
    
    @lru_cache(maxsize=cacheSize)
    def getFramenetScore(self, dataPoint):
        #sum of probabilities of encoding causality for words for different parts of speech
        #stem and lowercase
        for pos in self.framenetScores:
            if self.framenetScores[pos] is None:
                self.framenetScores[pos] = defaultdict(float)
                with open(os.path.join(os.path.join(FeatureExtractor.configPath,
                                                    'framenet'), pos)) as f:
                    for line in f:
                        p,word,count1,count2,score,entropy = line.split()
                        score = float(entropy)
                        if score > 0.0:
                            self.framenetScores[pos][self.sn.stem(word.lower())] \
                                += score

        #for now just do one aggregate score, also try different parts of speech
        #variations - only look at altlex or post altlex or entire sentence
        #           - look at previous sentence
        #           - combined score for all parts of speech
        #           - individual scores
        score = defaultdict(float)

        prevStart = 0
        prevEnd = dataPoint.prevSentenceLength
        altlexStart = prevEnd
        altlexEnd = prevEnd + dataPoint.altlexLength
        currStart = altlexEnd
        currEnd = altlexEnd + dataPoint.currSentenceLengthPostAltlex
        
        poses = dataPoint.getPrevPos() + dataPoint.getCurrPos()
        stems = dataPoint.getPrevStem() + dataPoint.getCurrStem()
        
        for start,end,fragment in ((altlexStart, altlexEnd, '_altlex'),
                                   (currStart, currEnd, '_curr_post'),
                                   (prevStart, prevEnd, '_prev')):
            #print(start, end, fragment)
            score[fragment] = 0
            score[fragment + '_anticausal'] = 0
            for i in range(start, end):
                pos = poses[i][:2].lower()
                stem = stems[i]
                #print(pos, stem)
                if pos in self.framenetScores and stem in self.framenetScores[pos]:
                    score[fragment] += self.framenetScores[pos][stem]
                pos += '_anticausal'
                if pos in self.framenetScores and stem in self.framenetScores[pos]:
                    score[fragment + '_anticausal'] += self.framenetScores[pos][stem]

        #print(score)
        return {self.functionFeatures[self.getFramenetScore] + fragment :
                score[fragment] for fragment in score}

    #syntactic features based on work by Pitler, Biran
    #also consider syntactic angles
    #production rules seems to cause overtraining for such a small dataset
    #what about just production rules within the altlex?
    @lru_cache(maxsize=cacheSize)
    def getProductionRules(self, dataPoint):
        '''return the nonlexical production rules for the tree'''
        tree = dataPoint.getCurrParse()
        return {self.functionFeatures[self.getProductionRules] + str(s) :
                True for s in tree.productions() if s.is_nonlexical()}

    @lru_cache(maxsize=cacheSize)
    def getAltlexProductionRules(self, dataPoint):
        altlexTree = extractSelfParse(dataPoint.getAltlex(),
                                      dataPoint.getCurrParse())
        if altlexTree is None:
            return {self.functionFeatures[self.getAltlexProductionRules] + ' None' :
                    True}
        else:
            return {self.functionFeatures[self.getAltlexProductionRules] + str(s) :
                    True for s in altlexTree.productions() if s.is_nonlexical()}
        
    #consider other things to do with siblings
    #use only the immediate right sibling, unless it is a punctuation or modifying phrase
    #print all the patterns of altlexes that we see (could use for bootstrapping)
    @lru_cache(maxsize=cacheSize)
    def getRightSiblings(self, dataPoint):
        tree = dataPoint.getCurrParse()
        altlex = dataPoint.getAltlex()
        if len(altlex):
            siblings = extractRightSiblings(altlex, tree)
        else:
            siblings = []
            
        #print(tree, siblings)
        return {self.functionFeatures[self.getRightSiblings] + s :
                True for s in siblings}

    #how about right sibling contains a VP or trace?

    @lru_cache(maxsize=cacheSize)
    def getSelfCategory(self, dataPoint):
        #could be none if altlex is not fully contained in a constituent
        altlex = dataPoint.getAltlex()
        if len(altlex):
            cat = extractSelfCategory(altlex,
                                      dataPoint.getCurrParse())
        else:
            cat = None

        return {self.functionFeatures[self.getSelfCategory] + str(cat) :
                True}
        
    @lru_cache(maxsize=cacheSize)
    def getParentCategory(self, dataPoint):
        #parent of self category
        #could be none if altlex is not fully contained in a constituent
        cat = extractParentCategory(dataPoint.getAltlex(),
                                    dataPoint.getCurrParse())

        return {self.functionFeatures[self.getParentCategory] + str(cat) :
                True}

    def _getVerbNetClasses(self, dataPoint, part, name):
        features = {}
        prevVerbs = dataPoint.getStemsForPos('V', part, 'lemmas')

        for verb in prevVerbs:
            for verbClass in self.vn.getClasses(verb):
                features[name + verbClass] = True

        if not len(features):
            features[name + 'None'] = True
            
        return features

    @lru_cache(maxsize=cacheSize)
    def getVerbNetClassesPrevious(self, dataPoint):
        return self._getVerbNetClasses(dataPoint,
                                       'previous',
                                       self.functionFeatures[self.getVerbNetClassesPrevious])
        
    @lru_cache(maxsize=cacheSize)
    def getVerbNetClassesCurrent(self, dataPoint):
        return self._getVerbNetClasses(dataPoint,
                                       'current',
                                       self.functionFeatures[self.getVerbNetClassesCurrent])

    @lru_cache(maxsize=cacheSize)
    def getVerbNetClassesAltlex(self, dataPoint):
        return self._getVerbNetClasses(dataPoint,
                                       'altlex',
                                       self.functionFeatures[self.getVerbNetClassesAltlex])    

    def _getThematicRoles(self, dataPoint, part, name):
        features = {}
        prevVerbs = dataPoint.getStemsForPos('V', part, 'lemmas')

        for verb in prevVerbs:
            for thematicRole in self.vn.getThematicRoles(verb):
                features[name + thematicRole] = True

        if not len(features):
            features[name + 'None'] = True
            
        return features

    @lru_cache(maxsize=cacheSize)
    def getThematicRolesPrevious(self, dataPoint):
        return self._getThematicRoles(dataPoint,
                                       'previous',
                                       self.functionFeatures[self.getThematicRolesPrevious])
        
    @lru_cache(maxsize=cacheSize)
    def getThematicRolesCurrent(self, dataPoint):
        return self._getThematicRoles(dataPoint,
                                       'current',
                                       self.functionFeatures[self.getThematicRolesCurrent])

    @lru_cache(maxsize=cacheSize)
    def getThematicRolesAltlex(self, dataPoint):
        return self._getThematicRoles(dataPoint,
                                       'altlex',
                                       self.functionFeatures[self.getThematicRolesAltlex])    

    @lru_cache(maxsize=cacheSize)
    def getVerbCluster(self, dataPoint):
        if self.verbClusters is None:
            self.verbClusters = {}
            with open(os.path.join(FeatureExtractor.configPath,
                                   'verb_clusters.json')) as f:
                vcs = json.load(f)
                for vc in vcs:
                    for v in vcs[vc]:
                        self.verbClusters[v] = vc
                    
        prevVerbs = dataPoint.getStemsForPos('V', 'previous', 'lemmas')
        features = {}
        for verb in prevVerbs:
            if verb in self.verbClusters:
                features[self.functionFeatures[self.getVerbCluster] + '_' + self.verbClusters[verb]] = True
            else:
                print("verb {} not here".format(verb))

        return features

    def getCoherence(self, dataPoint):
        features = {}
        m = dataPoint.coherence
        if m is not None:
            for index,value in enumerate(m):
                features['{}_{}'.format(self.functionFeatures[self.getCoherence],
                                        index)] = value

        return features

    def getNeuroCausal(self, dataPoint):
        if self.gensimModel is None:
            self.gensimModel = gensim.models.Word2Vec.load(self.gensimFile)
        if self.theanoModel is None:
            self.theanoModel = model(100, 2, self.gensimModel.layer1_size, 2)
            self.theanoModel.load(self.theanoDir)

        punct = {'!', '-', ',', '.', '?'}
        x1 = [self.gensimModel[w.lower()] for w in dataPoint.getPrevLemmas() if w.lower() in self.gensimModel and w not in punct]
        x2 = [self.gensimModel[w.lower()] for w in dataPoint.getCurrLemmasPostAltlex() if w.lower() in self.gensimModel and w not in punct]

        if not len(x1) or not len(x2):
            score = .5
        else:
            #print(len(x1), len(x2))
            score = self.theanoModel.classify(x1, x2)

        return {self.functionFeatures[self.getNeuroCausal]: score}

    def getNeuroDiscourse(self, dataPoint):
        if self.gensimModel is None:
            self.gensimModel = gensim.models.Word2Vec.load(self.gensimFile)
        if self.theanoModelDiscourse is None:
            self.theanoModelDiscourse = discourseModel(100, 102, self.gensimModel.layer1_size, 2)
            self.theanoModelDiscourse.load(self.theanoDirDiscourse)

        punct = {'!', '-', ',', '.', '?'}
        x1 = [self.gensimModel[w.lower()] for w in dataPoint.getPrevLemmas() if w.lower() in self.gensimModel and w not in punct]
        x2 = [self.gensimModel[w.lower()] for w in dataPoint.getCurrLemmasPostAltlex() if w.lower() in self.gensimModel and w not in punct]

        if not len(x1) or not len(x2):
            return {}
        else:
            scores = self.theanoModelDiscourse.score(x1, x2)

        return {self.functionFeatures[self.getNeuroDiscourse] + str(i): score for i,score in enumerate(scores)}

    def _getHeadVerbs(self, dataPoint):
        currHeadVerb = None
        for i,dep in enumerate(dataPoint.getCurrDependencies()):
            if dep is not None and dep[0] == 'root':
                currHeadVerb = dataPoint.getCurrLemmas()[i]
                break
        prevHeadVerb = None
        for i,dep in enumerate(dataPoint.getPrevDependencies()):
            if dep is not None and dep[0] == 'root':
                prevHeadVerb = dataPoint.getPrevLemmas()[i]
                break
            
        return currHeadVerb,prevHeadVerb

    def getHeadVerbCatPair(self, dataPoint):
        currHeadVerb,prevHeadVerb = self._getHeadVerbs(dataPoint)
        currCat = []
        if currHeadVerb is not None:
            currCat = self._getWordNetCat(['V'], [currHeadVerb], 'V')
        prevCat = []
        if prevHeadVerb is not None:
            prevCat = self._getWordNetCat(['V'], [prevHeadVerb], 'V')
        '''
        print(currHeadVerb, prevHeadVerb)
        print(currCat, prevCat)        
        print(dataPoint.getCurrWords())
        print(dataPoint.getCurrDependencies())
        print(dataPoint.getPrevWords())
        print(dataPoint.getPrevDependencies())
        '''
        if not len(currCat):
            currCat = None
        else:
            currCat = currCat[0]
        if not len(prevCat):
            prevCat = None
        else:
            prevCat = prevCat[0]

            
        features = {self.functionFeatures[self.getHeadVerbCatPair]: \
                    str(currCat) + '_' + str(prevCat)}
        #makes no difference, interestingly enough
        #features.update({self.functionFeatures[self.getHeadVerbCatPair]: \
        #                 str(prevCat) + '_' + str(currCat)})
        return features

    def getAllVerbCatPair(self, dataPoint):
        prevCats = self._getWordNetCat(dataPoint.getPrevPos(),
                                       dataPoint.getPrevLemmas(),
                                       'V')
        currCats = self._getWordNetCat(dataPoint.getCurrPos(),
                                       dataPoint.getCurrLemmas(),
                                       'V')
        if not len(prevCats) or not len(currCats):
            return {}
        features = {}
        for prevCat in prevCats:
            for currCat in currCats:
                features['{}_{}'.format(prevCat, currCat)] = True
        return features
                
    def getHeadVerbNetPair(self, dataPoint):
        currHeadVerb,prevHeadVerb = self._getHeadVerbs(dataPoint)
        features = {}

        '''
        print(currHeadVerb, prevHeadVerb)
        print(dataPoint.getCurrWords())
        print(dataPoint.getCurrDependencies())
        print(dataPoint.getPrevWords())
        print(dataPoint.getPrevDependencies())
        '''

        #TODO: if one has no verbs, wont return anything
        for verbClassCurr in self.vn.getClasses(currHeadVerb):
            for verbClassPrev in self.vn.getClasses(prevHeadVerb):
                #print(verbClassCurr, verbClassPrev)
                features[self.getHeadVerbNetPair] = '{}_{}'.format(verbClassCurr, verbClassPrev)

        return features
                                                  
    def getHeadVerbPair(self, dataPoint):
        currHeadVerb,prevHeadVerb = self._getHeadVerbs(dataPoint)
        '''
        '''
        return {self.functionFeatures[self.getHeadVerbPair]: \
                '{}_{}'.format(currHeadVerb,prevHeadVerb)}

    def preprocessKldFeatures(self, data):
        path = '/local/nlp/chidey/addDiscourse/washington/aligned_discourse/'
        latentFactors = joblib.load(path + '.model.Contingency.False.False.2.not_in_s1.100')
        featureSettings = {'bistems':True}
        #latentFactors = joblib.load(path + '.model.Contingency.True.True.2.not_in_s1.1000')
        #latentFactors = joblib.load(path + '.model.Contingency.True.True.2.in_s1.400')
        #featureSettings = {'bistems_sep':True, 'stem_pairs':True}
        
        X = []
        for i,dataPoint in enumerate(data):
            dp = DataPoint(dataPoint)
            X.append(self.addFeatures(dp, featureSettings))
        X_new = latentFactors.transform(X)
        self.latentFactorsLookup = {}
        for i,dataPoint in enumerate(data):
            dp = DataPoint(dataPoint)
            self.latentFactorsLookup[hash(dp)] = X_new[i]

    def kldLatentFactors(self, dataPoint):
        factors = self.latentFactorsLookup[hash(dataPoint)]
        return dict(zip(range(factors.shape[0]), factors))
    
    def addFeatures(self, dataPoint, featureSettings):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        features = {}
        for featureName in featureSettings:
            assert(featureName in self.validFeatures)
            if featureSettings[featureName]:
                features.update(self.validFeatures[featureName](dataPoint))
        return features

#verbnet features?
#lexpar features?
#adverbs are useful discourse relations but what resources are there for them?

#TODO: use dependency information to determine head verb
#ignore reporting verbs?
#if a sentence takes a clause as an object, use the head verb of the clause?
#use pairs of verbs
#use pairs of subjects?
#use pairs of verb states (wordnet)
#use pairs of verb classes (verbnet)

#chi2 or tf-idf feature selection (particularly for n-grams)
#scaling for other features

#separate reason and result into 2 separate classes?
