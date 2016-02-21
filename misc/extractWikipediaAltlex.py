from __future__ import print_function

import gzip
import json
import os
import sys
import time
import collections
import random

from nltk.stem import SnowballStemmer,PorterStemmer

from chnlp.misc import calcKLDivergence
from chnlp.misc import wiknet

from chnlp.utils import treeUtils
from chnlp.utils import wordUtils
from chnlp.utils import mlUtils
from chnlp.utils import dependencyUtils

from chnlp.altlex.config import Config
from chnlp.altlex.dataPoint import DataPoint, replaceNonAscii
from chnlp.altlex import featureExtractor
from chnlp.ml.sklearner import Sklearner

from chnlp.misc import reformatAnnotations
from chnlp.ml.gridSearch import HeldOutGridSearch

#from chnlp.word2vec import sentenceRepresentation

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

sn = SnowballStemmer('english') #PorterStemmer() #

featureSettings = {'altlex_length': True,
                   'final_reporting': True,
                   'final_time': True,
                   'head_verb_altlex' : True,
                   'head_verb_curr' : True,
                   'head_verb_prev' : True,
                   'verbnet_class_prev' : True,
                   'verbnet_class_curr' : True, #both seem to help
                   'framenet' : True}                  
customFeatureSettings = {#'in_causal': True,
                         #'in_notcausal': True,
                         'causal_kld': True,
                         'notcausal_kld': True,
                         'other_kld': True,
                         'has_proper_noun': True,
                         #'in_reason': True,
                         #'in_result': True,
                         'reason_kld': True,
                         'result_kld': True}
newFeatureSettings = {#'head_verb_net_pair': True,
                      'verbnet_class_altlex': True,
                      'noun_cat_altlex': True,
                      'noun_cat_curr': True,
                      'noun_cat_prev': True,
                      #'head_verb_cat_pair': True,
                      #'pos_indices': True
                      }
allFeatureSettings = {i:True for i in featureSettings.keys()+customFeatureSettings.keys()+newFeatureSettings.keys()}
bestFeatureSettings = {'causal_kld': True,
                       'notcausal_kld': True,
                       'reason_kld': True,
                       'result_kld': True,
                       'framenet' : True,
                       'final_reporting': True,
                       'final_time': True,
                       'verbnet_class_altlex': True,
                       'verbnet_class_prev' : True,
                       'verbnet_class_curr' : True,
                       'noun_cat_altlex': True,
                       'noun_cat_curr': True,
                       'noun_cat_prev': True,
                       'head_verb_altlex' : True,
                       'head_verb_curr' : True,
                       'head_verb_prev' : True,
                       }
labelLookup = {'notcausal' : 0,
               'causal' : 1,
               'other' : 2,
               'reason' : 3,
               'result' : 4}

def getMetaLabel(labels):
    if len(labels) == 0 or all(k[0] == 2 for k in labels):
        return 'other'
    elif all(k[0] in (1,2) for k in labels):
        return 'causal'
    elif all(k[0] in (0,2) for k in labels):
        return 'notcausal'
    elif all(k[0] in (3,2) for k in labels):
        return 'reason'
    elif all(k[0] in (4,2) for k in labels):
        return 'result'
    #elif not(any(k[0] for k in labels)):
    #    return 'notcausal'
    #elif all(k[0] for k in labels):
    #    return 'causal'
    return 'both'

def splitData(indices, numTest, ignoreBoth=False): #numCausal, numNonCausal, ignoreBoth=False):
    train, test, unclassified = set(), set(), set()

    for metaLabel in indices:
        if metaLabel in numTest:
            test |= set(indices[metaLabel][:numTest[metaLabel]])
            train |= set(indices[metaLabel][numTest[metaLabel]:])
        #if metaLabel == 'causal':
        #    test |= set(indices[metaLabel][:numCausal])
        #    train |= set(indices[metaLabel][numCausal:])
        #elif metaLabel == 'notcausal':
        #    test |= set(indices[metaLabel][:numNonCausal])
        #    train |= set(indices[metaLabel][numNonCausal:])
        elif metaLabel == 'other':
            unclassified |= set(indices[metaLabel])
        elif not ignoreBoth: # 'both'
            train |= set(indices[metaLabel])

    return train, test, unclassified

def iterData(labels, altlexes, pair):
    labelLookup = {j:i for i,j in labels}
    words = [[i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[0]['words'])],
              [i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[1]['words'])]]        

    lemmas = [[i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[0]['lemmas'])],
              [i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[1]['lemmas'])]]        
    pos = [wiknet.getLemmas(pair[0]['pos']),
           wiknet.getLemmas(pair[1]['pos'])]
    ner = [wiknet.getLemmas(pair[0]['ner']),
           wiknet.getLemmas(pair[1]['ner'])]
    stems = [[sn.stem(replaceNonAscii(i)) for i in lemmas[0]],
             [sn.stem(replaceNonAscii(i)) for i in lemmas[1]]]

    combinedDependencies = []
    for pairIndex,half in enumerate(pair):
        partialDependencies = []
        for depIndex,dep in enumerate(half['dep']):
            partialDependencies.append(dependencyUtils.tripleToList(dep, len(pair[pairIndex]['lemmas'][depIndex])))
        combinedDependencies.append(dependencyUtils.combineDependencies(*partialDependencies))
        
    #TODO: handle dependencies
    #if two sentences, don't need to really do anything (except remove altlex from deps)
    #if one sentence, call splitDependencies

    for index,altlex in enumerate(altlexes):
        for i,which in enumerate(altlex):
            newDependencies = dependencyUtils.splitDependencies(combinedDependencies[i], (which[0],which[1]))
            dataPoint = {'altlexLength': which[1]-which[0],
                         'orig_dep': pair[i]['dep'],
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
            label = labelLookup.get(index, 2)
            dp = DataPoint(dataPoint)
            yield dp, label

def getData(data, *indicesList):
    ret = [[] for i in indicesList]
    for datumId,(sentenceId,datum) in enumerate(data):
        index = [i for i,indices in enumerate(indicesList) if sentenceId in indices]
        #if len(index): print(index) 
        for i in index:
            ret[i].append((datumId,datum))
    return ret

class WeightedCounts:
    def __init__(self, weights=None):
        self.weights = weights
        if weights is None:
            self.weights = collections.defaultdict(dict)
            
    def __call__(self, rel, word):
        try:
            return self.weights[rel][word]
        except KeyError:
            return 1

    def makeWeights(self, pairedIterator, seedSet, base=1, lamda=1, update=False):
        for label,stems1,stems2,w in pairedIterator:
            if label in seedSet:
                for stem in (stems1 & seedSet[label]) | (stems2 & seedSet[label]):
                    if update or stem not in self.weights[label]:
                        self.weights[label][stem] = base
                if bool(stems1 & seedSet[label]) ^ bool(stems2 & seedSet[label]):
                    for stem in (stems2 - seedSet[label]) | (stems1 - seedSet[label]):
                        if update or stem not in self.weights[label]:                    
                            self.weights[label][stem] = base*2**-lamda
        
class PreprocessedIterator(calcKLDivergence.ParsedPairIterator):
    def __init__(self, indir, labels, altlexes, verbose=False):

        self.indir = indir
        self.labels = labels
        self.altlexes = altlexes
        self.verbose = verbose

    def iterLabels(self, indices=None):
        for index,pair in enumerate(calcKLDivergence.ParsedPairIterator.__iter__(self)):
            if indices is not None and index not in indices:
                continue
            labels = self.labels[index]
            altlexes = self.altlexes[index]
            yield index,labels,altlexes,pair

    def updateLabels(self, labelMap, classes, indices=None, verbose=False, n='all', balance=True):
        #if n is 'all', just take all of the smallest class and subsample the rest
        idLookup = labelMap
        labelMap = {}
        classCounts = collections.Counter(j[1] for j in idLookup.values())
        smallestClass = min(classCounts.values())
        print(smallestClass)

        if balance == True:
            balanceFactor = [1]*len(classes)
        else:
            balanceFactor = [None]*len(classes)
            for i in classes:
                if classCounts[i] == smallestClass:
                    balanceFactor[i] = 1
                else:
                    balanceFactor[i] = 1.*classCounts[i] / smallestClass
            
        if n == 'all':
            for i in classes:
                if classCounts[i] == smallestClass:
                    labelMap.update({j:k[1] for j,k in idLookup.items() if k[1] == i})
                else:
                    currentClassIds = filter(lambda x:x[1][1]==i, idLookup.items())
                    for j in range(smallestClass*balanceFactor[i]):
                        newIndex = int(random.random()*len(currentClassIds))
                        if newIndex % 2 == 0:
                            newIndex2 = newIndex
                        else:
                            newIndex2 = newIndex - 1

                        for index in ((newIndex, newIndex2)):
                            ix,k = currentClassIds.pop(index)
                            labelMap[ix] = k[1]
        else:
            #otherwise sort the points by the harmonic mean of their confidences and
            #take the n most confident points in each class
            for i in classes:
                #sort in descending order
                currentClassIds = sorted(filter(lambda x:x[1][1]==i,
                                                idLookup.items()),
                                         key=lambda x:x[1][0],
                                         reverse=True)
                labelMap.update({j:k[1] for j,k in currentClassIds[:int(balanceFactor[i]*n*2)]})

        print(len(labelMap))

        total = 0
        newLabelList = []
        newTrainingIds = set()
        newDatumIds = set()
        for sentenceId,labels,altlexes,pair in self.iterLabels(indices):
            if verbose and sentenceId % 10000 == 0:
                print(sentenceId)

            newLabels = labels
            if len(altlexes):
                words = [[i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[0]['words'])],
                         [i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[1]['words'])]]

            #new: only add if altlexes are different and they both agree
            for index,altlex in enumerate(altlexes):
                label1 = labelMap.get(total, None)
                label2 = labelMap.get(total+1, None)
                newLabel = filter(lambda x:x is not None, [label1, label2])
                total += 2
                #if words[0][altlex[0][0]:altlex[0][1]] == words[1][altlex[1][0]:altlex[1][1]]:
                #    continue
                if len(newLabel) == 2 and newLabel[0] == newLabel[1]:
                #if len(newLabel) and not (len(newLabel) == 2 and newLabel[0] != newLabel[1]):
                    newLabels.append([int(newLabel[0]), index])
                    newTrainingIds.add(sentenceId)
                    newDatumIds.update({total-2, total-1})
            newLabelList.append(newLabels)

        self.labels = newLabelList
        #also need the altlex that it is paired with
        
        #should we only add them if they are different altlexes and scored the same? undetermined
        
        #add new labels to unclassified data that has one of the known altlexes

        #now iterate through 
        return newTrainingIds, newDatumIds
    
    def iterAltlexes(self, indices=None):
        for index,labels,altlexes,pair in self.iterLabels(indices):
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
        
    def iterMetaLabels(self, indices=None):
        for labels,stems in self.iterAltlexes(indices):
            stems = [set(stems[0]), set(stems[1])]
            
            #then determine whether this is a training point and what type it is
            metaLabel = getMetaLabel(labels)
            
            yield metaLabel,stems[0],stems[1]
            yield metaLabel,stems[1],stems[0]

    def iterPairedLabels(self, indices=None, weighting=lambda x,y:1):
        for labels, stems in self.iterAltlexes(indices):
            labelLookup = {j:i for i,j in labels}
            for i in range(len(stems[0])):
                label = labelLookup.get(i, None)
                if label is None:
                    metaLabel = getMetaLabel([])
                else:
                    metaLabel = getMetaLabel([(label, i)])
                stems1 = {tuple(stems[0][i])}
                stems2 = {tuple(stems[1][i])}

                yield metaLabel,stems1,stems2,weighting
                yield metaLabel,stems2,stems1,weighting
                    
    def getIndices(self):
        indices = collections.defaultdict(list)
        for index,labels in enumerate(self.labels):
            metaLabel = getMetaLabel(labels)
            indices[metaLabel].append(index)
        return indices

    def iterAltlexPos(self, indices=None, verbose=False):
        foundIndices = set()
        for index,labels,altlexes,pair in self.iterLabels(indices):
            if verbose and index % 10000 == 0:
                print(index)

            for dp,label in iterData(labels, altlexes, pair):
                yield tuple(dp.getAltlexLemmatized())+tuple(dp.getAltlexPos()), label
            
            if indices:
                foundIndices.add(index)
                if len(foundIndices) == len(indices):
                    break
                    
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

def makeLabels(parsedPairsDir, initLabelsFile, alignments, seedSet):

    #if file of initial labels does not exist
    if not os.path.exists(initLabelsFile):
        pairIterator = calcKLDivergence.ParsedPairIterator(parsedPairsDir)
        
        labels = [] #list of tuples(label, altlexIndex)
        altlexes = [] #list of list of tuples ((sent0Start, sent0End), (sent1Start, sent1End)) ...

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
                for causalType in seedSet.keys():
                    for i,altlex in enumerate(alignedAltlex):
                        if any(seed.issubset(lemmas[i][altlex[0]:altlex[1]] + pos[i][altlex[0]:altlex[1]]) for seed in seedSet[causalType]):
                            label = labelLookup[causalType]
                if label is not None:
                    labelInfo.append([label, altlexIndex])
            labels.append(labelInfo)
            altlexes.append(alignedAltlexes)

            #TODO - also handle multi sentences - this should work fine i think?
            if pair[0]['lemmas']  != pair[1]['lemmas']:
                print(pair[0]['lemmas'], pair[1]['lemmas'], labelInfo, alignedAltlexes, alignment)

        #save labels AND aligned altlexes
        #labels will change at each iteration, aligned altlexes will not
        with gzip.open(initLabelsFile, 'w') as f:
            json.dump([labels, altlexes], f)
    else:
        print('loading labels...')
        with gzip.open(initLabelsFile) as f:
            labels, altlexes = json.load(f)

    return labels, altlexes

def addFeatures(dp,
                featureExtractor,
                deltaKLD,
                causalPhrases,
                features=None,
                sentenceEmbeddings=None,
                altFeatureSettings=None):
    
    altlex_ngram = tuple(dp.getAltlexLemmatized())
    altlex_pos = tuple(dp.getAltlexPos())

    if features is None:
        if altFeatureSettings is None:
            features = featureExtractor.addFeatures(dp, featureSettings)
        else:
            features = featureExtractor.addFeatures(dp, altFeatureSettings)
        features['has_proper_noun'] = any(i in ('NNP', 'NNPS') for i in altlex_pos)

    if altFeatureSettings is None and not set(features) & set(newFeatureSettings):
        features.update(featureExtractor.addFeatures(dp, newFeatureSettings))

    features['altlex'] = altlex_ngram + altlex_pos
    for key in deltaKLD:
        features[key + '_kld'] = deltaKLD[key].get(altlex_ngram + altlex_pos, 0)
    for key in causalPhrases:
        features['in_' + key] = altlex_ngram in causalPhrases[key]
            
    #add some latent semantic features (but only if the label is not other)
    #also no need to recalculate
    if sentenceEmbeddings is not None:
        if 'a_embedding' not in features and 'b_embedding' not in features:
            a_words = [i.lower() for i in dp.getPrevWords()]
            b_words = [i.lower() for i in dp.getCurrWordsPostAltlex()]
            print(a_words, altlex_ngram, b_words)
            features['a_embedding'] = sentenceEmbeddings.infer(a_words)
            features['b_embedding'] = sentenceEmbeddings.infer(b_words)

    return features

def makeDataset(pairIterator,
                deltaKLD,
                causalPhrases,
                featureExtractor,
                #tensor,
                indices=None,
                precalculated=None,
                sentenceEmbeddings=None,
                altFeatureSettings=None):

    if precalculated is None:
        dataset = []
    else:
        print('using precalculated features')
        dataset = precalculated
    total = 0

    for sentenceId,labels,altlexes,pair in pairIterator.iterLabels(indices):
        if sentenceId % 10000 == 0:
            print(sentenceId)
        
        for dp, label in iterData(labels, altlexes, pair):
            if precalculated is None:
                features = None
            else:
                features = dataset[total][1][0]                

            features = addFeatures(dp,
                                   featureExtractor,
                                   deltaKLD,
                                   causalPhrases,
                                   features,
                                   sentenceEmbeddings,
                                   altFeatureSettings=altFeatureSettings)

            if precalculated:
                dataset[total] = (sentenceId,(features, label))
            else:
                dataset.append((sentenceId,(features, label)))

            total += 1

    return dataset

if __name__ == '__main__':
    
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
    initLabelsFile = prefix + '_initLabels.json.gz'
    featuresFile = prefix + '_features.json.gz'
    classifierFile = prefix + '_classifier'
    testingFile = prefix + '_testing'
    wiki = '/proj/nlp/users/chidey/parallelwikis4.json.gz'
    model = '/local/nlp/chidey/model.wikipairs.doc2vec.pairwise.words'
    sclient = None #sentenceRepresentation.PairedSentenceEmbeddingsClient(wiki, model)
    
    seedSet = {'causal': [set(i) for i in wordUtils.causal_markers],
              'notcausal': [set(i) for i in wordUtils.noncausal_markers]}
    numTesting = {'causal': 100, 'notcausal': 1000}
    #seedSet = {'reason': [set(i) for i in wordUtils.reason_markers],
    #           'result': [set(i) for i in wordUtils.result_markers],
    #           'notcausal': [set(i) for i in wordUtils.noncausal_markers]}
    #numTesting = {'reason': 100, 'result': 100, 'notcausal': 2000}
    testing = None
    labeledDir = None
    if len(sys.argv) > 5:
        if len(sys.argv) > 6:
            with gzip.open(sys.argv[5]) as f:
                parses = json.load(f)
            labeledDir = sys.argv[6]
        else:
            with gzip.open(sys.argv[5]) as f:
                testing = json.load(f)
        numTesting = {'causal': 0, 'notcausal': 0}
        #numTesting = {'reason': 0, 'result': 0, 'notcausal': 0}

    labels, altlexes = makeLabels(sys.argv[1], initLabelsFile, alignments, seedSet)

    maxIterations = 10

    config = Config()
    #classifierType = config.classifiers['sgd']
    #classifierSettings = {u'penalty': u'elasticnet', u'alpha': 0.001, 'n_iter': 1000} #config.classifierSettings['grid_search_sgd']
    #classifierSettings = {u'penalty': u'elasticnet', u'alpha': 0.0001, 'n_iter': 1000}
    #classifier = Sklearner(classifierType(**classifierSettings))
    parameters = {"classifier": "sgd",
                  "verbose":True,
                  "n_jobs":1,
                  "parameters":{"n_iter":1000},
                  "searchParameters":{"penalty":["l2","elasticnet"],
                                      "alpha":[0.001, 0.0001, 0.00001, 0.000001]}}
    classifier = HeldOutGridSearch(**parameters)

    if False: #UNDO os.path.exists(featuresFile):
        print('loading features...')
        with gzip.open(featuresFile) as f:
            featureSet = json.load(f)
    else:
        featureSet = None

    pairIterator = PreprocessedIterator(sys.argv[1], labels, altlexes, verbose=False)
    indices = pairIterator.getIndices()
    #set aside a test/dev set that will remain unchanged at each iteration
    train, test, unclassified = splitData(indices,
                                          numTesting)
                                          #numCausalTesting,
                                          #numNonCausalTesting)
    print(len(train), len(test), len(unclassified))

    base=1
    lamda=0.5
    seedSetTuples = {'causal': wordUtils.causal_markers,
                     'notcausal': wordUtils.causal_markers}
    
    for iteration in range(maxIterations):
        #calculate KL divergence
        print(iteration)

        '''
        #TODO
        weightedCounts = WeightedCounts()
        weightedCounts.makeWeights(pairIterator.iterPairedLabels((train | unclassified) - test),
                                   seedSetTuples,
                                   base=base,
                                   lamda=base*lamda)
        base *= lamda
        '''
        if iteration == 0 or not featureSet: #UNDO 
            print('calculating kld at {}...'.format(time.time()))

            kldt = calcKLDivergence.main(pairIterator.iterPairedLabels((train | unclassified) - test,
                                                                       ), #TODO weighting=weightedCounts),
                                         withS1=False,
                                         prefix=prefix + str(iteration))

            deltaKLD = mlUtils.makeDeltaKLD(kldt, True)
            
            #calculate causal phrases from starting seeds
            print('calculating causal mappings at {}...'.format(time.time()))
            causalPhrases = {} #UNDO calcKLDivergence.getCausalPhrases(phrases['phrases'], seedSet, stem=False)
            #TODO: never updated the seed set
            
            #TODO: some kind of factorization (or maybe just use doc2vec)

        #add features
        print('adding features at {}...'.format(time.time()))

        featureSet = makeDataset(pairIterator,
                                 deltaKLD,
                                 causalPhrases,
                                 config.featureExtractor,
                                 precalculated=featureSet,
                                 sentenceEmbeddings=sclient)

        if False: #UNDO iteration == 0 and not os.path.exists(featuresFile):
            print('writing features to file...')
            with gzip.open(featuresFile, 'w') as f:
                json.dump(featureSet, f)

            trainSet = [(i[1][0], i[1][1]) for i in featureSet if i[1][1] != 2]
            with gzip.open('train.' + featuresFile, 'w') as f:
                json.dump(trainSet, f)
                
        print(time.time())
        print(len(featureSet))

        #train classifier on marked data points            
        training, _, remaining = getData(featureSet, train, test, unclassified)

        #TODO: update seed set tuples here
        '''
        reverseLabelLookup = {j:i for i,j in labelLookup.items()}
        for index,(features,label) in training:
            if label != 2:
                seedSetTuples[reverseLabelLookup[label]].add(features['altlex'])
        '''
        
        if labeledDir is None:
            if testing is None:
                testing = [({i:j for i,j in k[1][0].items() if i != 'altlex'},
                            k[1][1]) for k in _ if k[1][1] != 2]
            else:
                #TODO: recalculate KLD
                pass
        else:
            altlexes = set(tuple(i[1][0]['altlex']) for i in training if i[1][1] == 1)
            testingUnlabeled = reformatAnnotations.getCausalAnnotations(labeledDir, parses, altlexes)
            print('Num Testing: {}'.format(len(testingUnlabeled)))
            testing = []
            for datapoint in testingUnlabeled:
                label = datapoint['tag'] == 'causal'
                dp = DataPoint(datapoint)
                features = addFeatures(dp,
                                       config.featureExtractor,
                                       deltaKLD,
                                       causalPhrases,
                                       None,
                                       sclient)
                testing.append((features,label))
            with gzip.open('{}.{}.json.gz'.format(testingFile,
                                                  iteration), 'w') as f:
                json.dump(testing, f)

        #knownAltlexes = set(tuple(i[1][0]['altlex']) for i in training if i[1][1] != 2)
        knownAltlexes = {}
        for i in seedSet:
            knownAltlexes[labelLookup[i]] = collections.Counter(tuple(j[1][0]['altlex']) for j in training if j[1][1] == labelLookup[i])
            
        training = [({i:j for i,j in k[1][0].items() if i != 'altlex'},
                     k[1][1]) for k in training if k[1][1] != 2] # and 'NNP' not in k[1][0]['altlex'] and 'NNPS' not in k[1][0]['altlex']]

        data = []
        for index,dataset in enumerate((training, testing)):
            data.append([])
            for datum,label in dataset:
                newDatum = {i:j for i,j in datum.items() if any(i.startswith(f) for f in bestFeatureSettings)}
                data[-1].append((newDatum,label))

        X, y = zip(*data[0])
        classifier.setTuning(data[1])
        classifier.fit_transform(X, y)
        classifier.save('{}.{}'.format(prefix, iteration))
        X,y = zip(*data[1])
        accuracy, precision, recall, f_score, predictions = classifier.metrics(X, y)
        print(accuracy, precision, recall, f_score)
        for i in range(len(precision)):
            print('Class {}'.format(i))
            classifier.printResults(accuracy, precision[i], recall[i])

        idLookup = {}
        for i in knownAltlexes:
            #first find any points where we know the altlex
            possibleNewTraining = [(datum[0],
                                    datumId) for datumId,datum in remaining if tuple(datum[0]['altlex']) in knownAltlexes[i]]
            #then add any points that are matched with a known altlex
            matchedIndices = {i[1] for i in possibleNewTraining}
            for p in possibleNewTraining:
                if p[1] % 2 == 0:
                    matchedIndices.add(p[1]+1)
                else:
                    matchedIndices.add(p[1]-1)
            possibleNewTraining = [(datum[0],
                                    datumId) for datumId,datum in remaining if datumId in matchedIndices]
        
            possibleNewTraining,ids = zip(*possibleNewTraining)
            y_predict = classifier.predict(classifier.transform(possibleNewTraining))
            y_scores = classifier.decision_function(classifier.transform(possibleNewTraining))

            #only include if occurs in desired class minimum X times and y_predict agrees with this class
            #TODO: or only include based on percentage of appearances in current class or other classes

            minCount = 1
            for j in range(0,len(possibleNewTraining),2):
                if knownAltlexes[i][tuple(possibleNewTraining[j]['altlex'])] >= minCount or knownAltlexes[i][tuple(possibleNewTraining[j+1]['altlex'])] >= minCount:
                    if y_predict[j] == i and y_predict[j+1] == i:
                        harmonicMean = abs((2*y_scores[j]*y_scores[j+1])/(y_scores[j]+y_scores[j+1]))
                        idLookup[ids[j]] = harmonicMean,i
                        idLookup[ids[j+1]] = harmonicMean,i
                
            #newTrainingIds,newDatumIds = pairIterator.updateLabels(dict(zip(ids, y_predict)))

        #TODO: manage ratio of added to not added
                    
        #newTrainingIds,newDatumIds = pairIterator.updateLabels(idLookup)
        print(len(idLookup))
        newTrainingIds,newDatumIds = pairIterator.updateLabels(idLookup,
                                                               classes=knownAltlexes,
                                                               n=500,
                                                               balance=False) #UNDO
        
        #remove from unclassifed and add to train
        train |= newTrainingIds
        unclassified -= newTrainingIds

        print(len(newTrainingIds), len(newDatumIds))

        for i in knownAltlexes:
            print('*'*79)           
            print(i)
            newAltlexes = collections.Counter(tuple(featureSet[j][1][0]['altlex']) for j in idLookup if j in newDatumIds and idLookup[j][1] == i)
            print(len(newAltlexes), sum(newAltlexes.values()))
            print('*'*79)
            for j in sorted(newAltlexes.items(), key=lambda x:x[1]): #UNDO [-30:]:
                print(j)
            print('*'*79)
        
    training, _, _ = getData(featureSet, train, test, unclassified)
    with gzip.open(prefix + '_finalTraining.json.gz', 'w') as f:
        json.dump(training, f)
    #with gzip.open('_finalLabels.json.gz', 'w') as f:
    #    json.dump([pairIterator.labels, pairIterator.altlexes], f)
