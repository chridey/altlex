import itertools
from collections import defaultdict

import nltk

from altlex.featureExtraction.dataPoint import DataPoint
from altlex.featureExtraction import config

from altlex.semantics.wordNetManager import WordNetManager
from altlex.semantics.verbNetManager import VerbNetManager
from altlex.semantics.frameNetManager import FrameNetManager

from altlex.utils.dependencyUtils import getRoot,getEventAndArguments

from altlex.ml.kldManager import KLDManager

class FeatureExtractor:
    
    def __init__(self,
                 settings=config.defaultConfig,
                 verbose=False):
        self.config = config.Config(settings)
        
        self.wn = WordNetManager()
        self.vn = VerbNetManager()
        if self.config.framenetSettings is not None:
            self.fn = FrameNetManager(verbose=verbose,
                                      **self.config.framenetSettings)
        else:
            self.fn = FrameNetManager(verbose=verbose)

        if self.config.KLDSettings is not None:
            self.kldm = KLDManager(verbose=verbose,
                                   **self.config.KLDSettings)
        else:
            self.kldm = KLDManager(verbose=verbose)
            
                              #lexical features
        self.validFeatures = {'unistems' : self.getUnistems,
                              'bistems' : self.getBistems,
                              'unistems_sep' : self.getUnistemsSep,
                              'bistems_sep' : self.getBistemsSep,
                              'connective': self.getConnective,
                              'connective_lemmas_pos': self.getConnectiveLemmasPos,
                              'connective_unistems': self.getConnectiveUnistems,
                              'connective_bistems': self.getConnectiveBistems,
                              
                              #word pair features
                              'stem_pairs' : self.getStemPairs,
                              'all_stem_pairs' : self.getAllStemPairs,

                              #length features
                              'altlex_length': self.getAltlexLength,
                              'curr_length': self.getCurrLength,
                              'prev_length': self.getPrevLength,
                              'curr_length_post_altlex': self.getCurrPostLength,

                              #kld features derived from parallel corpus
                              'kld_score': self.getKLDScore,
                              
                              #score semantic features
                              'framenet' : self.getFramenetScore,

                              #resource semantic features
                              'head_word_cat_curr': self.getHeadWordCatCurr,
                              'head_word_cat_prev': self.getHeadWordCatPrev,
                              'head_word_cat_altlex': self.getHeadWordCatAltlex,
                              'head_word_verbnet_curr': self.getHeadWordVerbNetCurr,
                              'head_word_verbnet_prev': self.getHeadWordVerbNetPrev,
                              'head_word_verbnet_altlex': self.getHeadWordVerbNetAltlex,
                              'arguments_cat_curr': self.getArgumentsCatCurr,
                              'arguments_cat_prev': self.getArgumentsCatPrev,
                              'arguments_verbnet_curr': self.getArgumentsVerbNetCurr,
                              'arguments_verbnet_prev': self.getArgumentsVerbNetPrev,

                              #interaction features
                              }

        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

    #
    #lexical features
    #
    def getConnective(self, dataPoint):
        return {'connective_' + '_'.join(dataPoint.getAltlex()): True}

    def getConnectiveLemmasPos(self, dataPoint):
        return {'connective_' + '_'.join(dataPoint.getAltlexLemmasAndPos()): True}

    def getConnectiveUnistems(self, dataPoint):
        ret = {}
        for stem in dataPoint.getAltlexStem():
            ret['connective_unistem_' + stem] = True
        return ret
    
    def getConnectiveBistems(self, dataPoint):
        ret = {}
        for stem in nltk.bigrams(dataPoint.getAltlexStem()):
            ret['connective_unistem_' + '_'.join(stem)] = True
        return ret
    
    def _getNstems(self, dataPoint, n, prefix=False):
        if prefix:
            first = '1_'
            second = '2_'
        else:
            first = ''
            second = ''

        if n == 1:
            prev = [first+i for i in dataPoint.getPrevStem()]
            curr = [second+i for i in dataPoint.getCurrStem()]
        elif n == 2:
            prev = ['_'.join(i) for i in nltk.bigrams([first+i for i in dataPoint.getPrevStem()])]
            curr = ['_'.join(i) for i in nltk.bigrams([first+i for i in dataPoint.getCurrStem()])]

        return dict(zip(prev+curr, [True]*(len(prev)+len(curr))))
    
    def getUnistems(self, dataPoint):
        return self._getNstems(dataPoint, 1)

    def getBistems(self, dataPoint):
        return self._getNstems(dataPoint, 2)

    def getUnistemsSep(self, dataPoint):
        return self._getNstems(dataPoint, 1, True)

    def getBistemsSep(self, dataPoint):
        return self._getNstems(dataPoint, 2, True)

    #
    #word pair features
    #
    
    def getStemPairs(self, dataPoint):
        features = {}

        #only include those with connective
        pairs = list(itertools.product(['1_'+i for i in dataPoint.getPrevStem()],
                                       ['2_'+i for i in dataPoint.getCurrStem()]))
        return dict(zip(pairs, [True]*len(pairs)))

    def getAllStemPairs(self, dataPoint):
        pairs = list(itertools.permutations(dataPoint.getPrevStem()+dataPoint.getCurrStem(), 2))
        return dict(zip(pairs, [True]*len(pairs)))

    #
    #length features
    #
    
    def getAltlexLength(self, dataPoint):
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.altlexLength}

    def getCurrLength(self, dataPoint):
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.currSentenceLength}

    def getPrevLength(self, dataPoint):
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.prevSentenceLength}

    def getCurrPostLength(self, dataPoint):
        return {self.functionFeatures[self.getAltlexLength]:
                dataPoint.currSentenceLengthPostAltlex}

    #
    #parallel corpus features
    #
    def getKLDScore(self, dataPoint):
        scores = self.kldm.score(dataPoint.getAltlexLemmatized(),
                                 dataPoint.getAltlexPos())
        return {self.functionFeatures[self.getKLDScore] + '_' + key : scores[key] for key in scores}

    #
    #semantic features
    #

    def getFramenetScore(self, dataPoint):
        #sum of probabilities of encoding causality for words for different parts of speech
        #stem and lowercase
        
        #for now just do one aggregate score, also try different parts of speech
        #variations - only look at altlex or post altlex or entire sentence
        #           - look at previous sentence
        #           - combined score for all parts of speech
        #           - individual scores

        score = {}

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

            score.update(self.fn.score(stems[start:end],
                                       poses[start:end],
                                       suffix=fragment))

        return {self.functionFeatures[self.getFramenetScore] + fragment :
                score[fragment] for fragment in score}

    def _getHeadWordCat(self, function, root, lemmas, pos, ner):
        if root is not None and pos[root][0] in {'V', 'N', 'J', 'R'}:
            category = self.wn.wordCategory(lemmas[root], pos[root])
            if category is None and pos[root][0] == 'N' and ner[root] != "O":
                category = 'noun.' + ner[root].lower()
        else:
            category = None

        return {self.functionFeatures[function] + '_' + str(category): True}

    def getHeadWordCatCurr(self, dataPoint):
        root = getRoot(dataPoint.getCurrDependencies())
        return self._getHeadWordCat(self.getHeadWordCatCurr,
                                    root,
                                    dataPoint.getCurrLemmasPostAltlex(),
                                    dataPoint.getCurrPosPostAltlex(),
                                    dataPoint.getCurrNerPostAltlex())

    def getHeadWordCatPrev(self, dataPoint):
        root = getRoot(dataPoint.getPrevDependencies())
        return self._getHeadWordCat(self.getHeadWordCatPrev,
                                    root,
                                    dataPoint.getPrevLemmas(),
                                    dataPoint.getPrevPos(),
                                    dataPoint.getPrevNer())

    def getHeadWordCatAltlex(self, dataPoint):
        root = getRoot(dataPoint.getAltlexDependencies())
        return self._getHeadWordCat(self.getHeadWordCatAltlex,
                                    root,
                                    dataPoint.getAltlexLemmatized(),
                                    dataPoint.getAltlexPos(),
                                    dataPoint.getAltlexNer())

    def _getHeadWordVerbNet(self, function, root, lemmas, pos):
        ret = {}
        if root is not None and pos[root][0] == 'V':
            for verbClass in self.vn.getClasses(lemmas[root]):
                ret [self.functionFeatures[function] + '_' + verbClass] = True

        if not len(ret):
            return {self.functionFeatures[function] + '_' + str(None): True}
        return ret
    
    def getHeadWordVerbNetCurr(self, dataPoint):
        root = getRoot(dataPoint.getCurrDependencies())
        return self._getHeadWordVerbNet(self.getHeadWordVerbNetCurr,
                                        root,
                                        dataPoint.getCurrLemmasPostAltlex(),
                                        dataPoint.getCurrPosPostAltlex())
    
    def getHeadWordVerbNetPrev(self, dataPoint):
        root = getRoot(dataPoint.getPrevDependencies())
        return self._getHeadWordVerbNet(self.getHeadWordVerbNetPrev, root, dataPoint.getPrevLemmas(), dataPoint.getPrevPos())

    def getHeadWordVerbNetAltlex(self, dataPoint):
        root = getRoot(dataPoint.getAltlexDependencies())
        return self._getHeadWordVerbNet(self.getHeadWordVerbNetAltlex,
                                        root,
                                        dataPoint.getAltlexLemmatized(),
                                        dataPoint.getAltlexPos())

    def _getArgumentsCat(self, function, arguments, lemmas, pos, ner):
        ret = {}
        for argument in arguments:
            #limit to only content words
            if pos[argument][0] in {'V', 'N', 'J', 'R'}:
                category = self.wn.wordCategory(lemmas[argument], pos[argument])
                if category is None and pos[argument][0] == 'N' and ner[argument] != "O":
                    category = 'noun.' + ner[argument].lower()

                ret[self.functionFeatures[function] + '_' + str(category)] = True
                    
        if not len(ret):
            return {self.functionFeatures[function] + '_' + str(None) : True}
        return ret            

    def getArgumentsCatCurr(self, dataPoint):
        root,arguments = getEventAndArguments(dataPoint.getCurrDependencies())
        return self._getArgumentsCat(self.getArgumentsCatCurr,
                                     arguments,
                                     dataPoint.getCurrLemmasPostAltlex(),
                                     dataPoint.getCurrPosPostAltlex(),
                                     dataPoint.getCurrNerPostAltlex())
    
    def getArgumentsCatPrev(self, dataPoint):
        root,arguments = getEventAndArguments(dataPoint.getPrevDependencies())
        return self._getArgumentsCat(self.getArgumentsCatPrev,
                                     arguments,
                                     dataPoint.getPrevLemmas(),
                                     dataPoint.getPrevPos(),
                                     dataPoint.getPrevNer())

    def _getArgumentsVerbNet(self, function, arguments, lemmas, pos):
        ret = {}
        for argument in arguments:
            if pos[argument][0] == 'V':
                for verbClass in self.vn.getClasses(lemmas[argument]):
                    ret [self.functionFeatures[function] + '_' + verbClass] = True
        if not len(ret):
            return {self.functionFeatures[function] + '_' + str(None): True}
        return ret

    def getArgumentsVerbNetCurr(self, dataPoint):
        root,arguments = getEventAndArguments(dataPoint.getCurrDependencies())
        return self._getArgumentsVerbNet(self.getArgumentsVerbNetCurr,
                                         arguments,
                                         dataPoint.getCurrLemmasPostAltlex(),
                                         dataPoint.getCurrPosPostAltlex())
    
    def getArgumentsVerbNetPrev(self, dataPoint):
        root,arguments = getEventAndArguments(dataPoint.getPrevDependencies())
        return self._getArgumentsVerbNet(self.getArgumentsVerbNetPrev,
                                         arguments,
                                         dataPoint.getPrevLemmas(),
                                         dataPoint.getPrevPos())

    ################
    #done with features
    
    def addFeatures(self, dataPoint, featureSettings=None):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        if featureSettings is None:
            featureSettings = self.config.featureSettings
            
        features = {}
        for featureName in featureSettings:
            assert(featureName in self.validFeatures)
            if featureSettings[featureName]:
                features.update(self.validFeatures[featureName](dataPoint))
        return features

#make interaction features
#combine everything that matches pattern a with pattern b
def makeInteractionFeatures(features, pattern1, pattern2):
    new_features = {}
    for i in itertools.product(filter(lambda x:pattern1 in x, features.keys()),
                               filter(lambda x:pattern2 in x, features.keys())):
        new_features['_'.join(i)] = True
    return new_features

def filterFeatures(features, patterns=None, antipatterns=None):
    new_features = {}
    for feature in features:
        if (patterns is None or any(pattern in feature for pattern in patterns)) and (antipatterns is None or not any(antipattern in feature for antipattern in antipatterns)):
            new_features[feature] = features[feature]

    return new_features

def modifyFeatureSet(features, include=None, ablate=None, interaction=None):
    if include:
        features = filterFeatures(features,
                                  include.split(','),
                                  None)
        
    if ablate:
        features = filterFeatures(features,
                                  None,
                                  ablate.split(','))
                
    if interaction:
        filtered_features = filterFeatures(features,
                                           interaction['include'],
                                           interaction['ablate'])
                                                                    
        interaction_features = makeInteractionFeatures(filtered_features,
                                                       interaction['first'],
                                                       interaction['second'])
        features.update(interaction_features)

    return features

def createModifiedDataset(dataset, include=None, ablate=None, interaction=None):
    for data in dataset:
        data.features = modifyFeatureSet(data.features, include, ablate, interaction)
    return dataset
