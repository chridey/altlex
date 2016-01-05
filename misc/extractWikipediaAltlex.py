from __future__ import print_function

import gzip
import json
import os
import sys
import time
import collections

from nltk.stem import SnowballStemmer

from chnlp.misc import calcKLDivergence
from chnlp.misc import wiknet

from chnlp.utils import treeUtils

from chnlp.altlex.config import Config
from chnlp.altlex.dataPoint import DataPoint, replaceNonAscii
from chnlp.altlex import featureExtractor
from chnlp.ml.sklearner import Sklearner

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

sn = SnowballStemmer('english')
featureSettings = {'altlex_length': True,
                   'final_reporting': True,
                   'final_time': True,
                   'head_verb_altlex' : True,
                   'head_verb_curr' : True,
                   'head_verb_prev' : True,
                   'verbnet_class_prev' : True,
                   'verbnet_class_curr' : True, #both seem to help
                   'framenet' : True}                  
customFeatureSettings = {'in_causal': True,
                         'in_notcausal': True,
                         'causal_kld': True,
                         'notcausal_kld': True,
                         'other_kld': True,
                         'has_proper_noun': True}

allFeatureSettings = {i:True for i in featureSettings.keys()+customFeatureSettings.keys()}

def getMetaLabel(labels):
    if len(labels) == 0:
        return 'other'
    elif not(any(k[0] for k in labels)):
        return 'notcausal'
    elif all(k[0] for k in labels):
        return 'causal'
    return 'both'

def splitData(indices, numCausal, numNonCausal, ignoreBoth=False):
    train, test, unclassified = set(), set(), set()

    for metaLabel in indices:
        if metaLabel == 'causal':
            test |= set(indices[metaLabel][:numCausal])
            train |= set(indices[metaLabel][numCausal:])
        elif metaLabel == 'notcausal':
            test |= set(indices[metaLabel][:numNonCausal])
            train |= set(indices[metaLabel][numNonCausal:])
        elif metaLabel == 'other':
            unclassified |= set(indices[metaLabel])
        elif not ignoreBoth: # 'both'
            train |= set(indices[metaLabel])

    return train, test, unclassified

def iterData(labels, altlexes, pair):
    labelLookup = {j:i for i,j in labels}
    lemmas = [[i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[0]['lemmas'])],
              [i.lower().encode('utf-8') for i in wiknet.getLemmas(pair[1]['lemmas'])]]        
    pos = [wiknet.getLemmas(pair[0]['pos']),
           wiknet.getLemmas(pair[1]['pos'])]
    stems = [[sn.stem(replaceNonAscii(i)) for i in lemmas[0]],
             [sn.stem(replaceNonAscii(i)) for i in lemmas[1]]]
    for index,altlex in enumerate(altlexes):
        for i,which in enumerate(altlex):

            dataPoint = {'altlexLength': which[1]-which[0],
                         'sentences': [{
                             'lemmas': lemmas[i][which[0]:],
                             'words': lemmas[i][which[0]:],
                             'stems': stems[i][which[0]:],
                             'pos': pos[i][which[0]:]
                             },
                                       {
                                           'lemmas': lemmas[i][:which[0]],
                                           'words': lemmas[i][:which[0]],
                                           'stems': stems[i][:which[0]],
                                           'pos': pos[i][:which[0]]
                                           }]}
            label = labelLookup.get(index, 2)
            dp = DataPoint(dataPoint)
            yield dp, label

def getData(data, *indicesList):
    ret = [[] for i in indicesList]
    print(ret)
    for datumId,(sentenceId,datum) in enumerate(data):
        index = [i for i,indices in enumerate(indicesList) if sentenceId in indices]
        #if len(index): print(index) 
        for i in index:
            ret[i].append((datumId,datum))
    return ret

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

    def updateLabels(self, labelMap, indices=None, verbose=False):
        total = 0
        newLabelList = []
        newTrainingIds = set()
        for sentenceId,labels,altlexes,pair in self.iterLabels(indices):
            if verbose and sentenceId % 10000 == 0:
                print(sentenceId)

            newLabels = labels
            for index,altlex in enumerate(altlexes):
                label1 = labelMap.get(total, None)
                label2 = labelMap.get(total+1, None)
                newLabel = filter(lambda x:x is not None, [label1, label2])
                if len(newLabel) and not (len(newLabel) == 2 and newLabel[0] != newLabel[1]):
                    newLabels.append([int(newLabel[0]), index])
                    newTrainingIds.add(sentenceId)
                total += 2
            newLabelList.append(newLabels)

        self.labels = newLabelList
        #also need the altlex that it is paired with
        
        #should we only add them if they are different altlexes and scored the same? undetermined
        
        #add new labels to unclassified data that has one of the known altlexes

        #now iterate through 
        return newTrainingIds
    
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

    def iterPairedLabels(self, indices=None):
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
                yield metaLabel,stems1,stems2
                yield metaLabel,stems2,stems1

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
                yield tuple(dp.getAltlex())+tuple(dp.getAltlexPos()), label
            
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

def makeDataset(pairIterator,
                deltaKLD,
                causalPhrases,
                featureExtractor,
                #tensor,
                indices=None,
                precalculated=None):

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
            altlex_ngram = tuple(dp.getAltlex())
            altlex_pos = tuple(dp.getAltlexPos())

            if precalculated is None:
                features = featureExtractor.addFeatures(dp, featureSettings)
            else:
                features = dataset[total][1][0]

            features['altlex'] = altlex_ngram + altlex_pos
            features['causal_kld'] = deltaKLD['causal'].get(altlex_ngram + altlex_pos, 0)
            features['notcausal_kld'] = deltaKLD['notcausal'].get(altlex_ngram + altlex_pos, 0)
            features['other_kld'] = deltaKLD['other'].get(altlex_ngram + altlex_pos, 0)
            features['in_causal'] = altlex_ngram in causalPhrases['causal']
            features['in_notcausal'] = altlex_ngram in causalPhrases['notcausal']
            features['has_proper_noun'] = any(i in ('NNP', 'NNPS') for i in altlex_pos)
            #TODO: add other features

            if precalculated:
                dataset[total] = (sentenceId,(features, label))
            else:
                dataset.append((sentenceId,(features, label)))

            total += 1
            
    return dataset

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

    maxIterations = 10
    numCausalTesting = 100
    numNonCausalTesting = 1000
    config = Config()
    classifierType = config.classifiers['grid_search_sgd']
    classifierSettings = config.classifierSettings['grid_search_sgd']
    classifier = Sklearner(classifierType(**classifierSettings))
    if os.path.exists('features.json.gz'):
        print('loading features...')
        with gzip.open('features.json.gz') as f:
            featureSet = json.load(f)
    else:
        featureSet = None

    pairIterator = PreprocessedIterator(sys.argv[1], labels, altlexes, verbose=False)
    indices = pairIterator.getIndices()
    #set aside a test/dev set that will remain unchanged at each iteration
    train, test, unclassified = splitData(indices,
                                          numCausalTesting,
                                          numNonCausalTesting)
    print(len(train), len(test), len(unclassified))
    
    for iteration in range(maxIterations):
        #calculate KL divergence
        print(iteration)

        if iteration > 0 or not featureSet:
            print('calculating kld at {}...'.format(time.time()))

            kldt = calcKLDivergence.main(pairIterator.iterPairedLabels((train | unclassified) - test),
                                         withS1=False,
                                         prefix=prefix + str(i))    
            deltaKLD = collections.defaultdict(dict)

            for phraseType in kldt.keys():
                topKLD = kldt[phraseType][1].topKLD()
                for kld in topKLD:
                    if kld[1] > kld[2]:
                        score = kld[3]
                    else:
                        score = -kld[3]
                    deltaKLD[phraseType][kld[0]] = score
            for q in deltaKLD:
                print(q, len(deltaKLD[q]))
            
            #calculate causal phrases from starting seeds
            print('calculating causal mappings at {}...'.format(time.time()))
            causalPhrases = calcKLDivergence.getCausalPhrases(phrases['phrases'], seedSet, stem=False)
            #TODO: some kind of factorization

            #add features
            print('adding features at {}...'.format(time.time()))
            featureSet = makeDataset(pairIterator,
                                     deltaKLD,
                                     causalPhrases,
                                     config.featureExtractor,
                                     precalculated=featureSet)
        print(time.time())
        print(len(featureSet))


        #train classifier on marked data points            
        training, testing, remaining = getData(featureSet, train, test, unclassified)

        knownAltlexes = set(tuple(i[1][0]['altlex']) for i in training if i[1][1] != 2)
        training = [({i:j for i,j in k[1][0].items() if i != 'altlex'},
                     k[1][1]) for k in training if k[1][1] != 2]
        testing = [({i:j for i,j in k[1][0].items() if i != 'altlex'},
                     k[1][1]) for k in testing if k[1][1] != 2]        

        X, y = zip(*training)        
        classifier.fit_transform(X, y)
        X,y = zip(*testing)
        accuracy, precision, recall, f_score, predictions = classifier.metrics(X, y)
        print(accuracy, precision, recall, f_score)
        classifier.printResults(accuracy, precision[1], recall[1])

        possibleNewTraining = [({i:j for i,j in datum[0].items() if i != 'altlex'},
                                datumId) for datumId,datum in remaining if tuple(datum[0]['altlex']) in knownAltlexes]
        possibleNewTraining,ids = zip(*possibleNewTraining)
        y_predict = classifier.predict(classifier.transform(possibleNewTraining))
        newTrainingIds = pairIterator.updateLabels(dict(zip(ids, y_predict)))
        
        #remove from unclassifed and add to train
        train |= newTrainingIds
        unclassified -= newTrainingIds

        newCausalAltlexes = collections.Counter(tuple(featureSet[i][1][0]['altlex']) for i,j in zip(ids,y_predict) if j==1)
        newNonCausalAltlexes = collections.Counter(tuple(featureSet[i][1][0]['altlex']) for i,j in zip(ids,y_predict) if j==0)
        print(len(newCausalAltlexes), sum(newCausalAltlexes.values()))
        for i in sorted(newCausalAltlexes.items(), key=lambda x:x[1])[-30:]:
            print(i)
        print('*'*79)
        print(len(newNonCausalAltlexes), sum(newNonCausalAltlexes.values()))
        for i in sorted(newNonCausalAltlexes.items(), key=lambda x:x[1])[-30:]:
            print(i)
        print('*'*79)
        print(len(newTrainingIds))
        
        if iteration == 0 and not featureSet:
            with gzip.open('features.json.gz', 'w') as f:
                json.dump(featureSet, f)
            #with gzip.open('train.json.gz', 'w') as f:
            #    json.dump(training, f)
            #with gzip.open('test.json.gz', 'w') as f:
            #    json.dump(testing, f)
