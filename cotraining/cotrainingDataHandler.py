import json
import collections

from chnlp.utils.utils import splitData, sampleDataWithoutReplacement, iterFolds

class CotrainingDataHandler:
    def __init__(self,
                 untaggedLimit,
                 untaggedSampleSize):
        self.untaggedLimit = untaggedLimit
        self.untaggedSampleSize = untaggedSampleSize

        self.taggedTypes = {'training',
                            'testing'}
        self.untaggedTypes = {'cotraining',
                              'sampling'}
        
    def makeDataset(self,
                    taggedData,
                    untaggedData,
                    numFolds,
                    featureSubsets,
                    makeDataset,
                    config):
        #self.taggedData = taggedData
        #self.untaggedData = untaggedData
        #self.makeDataset = makeDataset
        #self.featureExtractor = featureExtractor
        self.numFolds = numFolds
        self.featureSubsets = featureSubsets

        self.taggedData = collections.defaultdict(list)
        self.untaggedData = {}
        
        #first create the metadata and split the datasets for the tagged data
        for index,featureSubset in enumerate(featureSubsets):
            positiveSet, negativeSet = makeDataset(taggedData,
                                                   config.featureExtractor,
                                                   config.featureSubset(*featureSubset))
            evaluation, testing = splitData(positiveSet, negativeSet)

            #add these to some lookup table, blah blah
            for training, testing in iterFolds(evaluation,
                                               n_folds=self.numFolds):
                self.taggedData[index].append({'training': training,
                                               'testing': testing})

        #now randomly get U examples from the untagged data
        for featureSubset in featureSubsets:
            emptySet, unknownSet = makeDataset(untaggedData,
                                               config.featureExtractor,
                                               config.featureSubset(*featureSubset),
                                               max=self.untaggedLimit)
                                               
            #sample here
            startingData, remainingData = sampleDataWithoutReplacement(unknownSet,
                                                                       self.untaggedSampleSize)
            self.untaggedData[index] = {'cotraining': startingData,
                                        'sampling': remainingData}

    def writeJSON(self, outfilename):
        #save num folds, feature subsets, tagged data, untagged data
        js = {'numFolds' : self.numFolds,
              'featureSubsets' : self.featureSubsets,
              'taggedData' : self.taggedData,
              'untaggedData' : self.untaggedData}
        with open(outfilename, 'w') as f:
            json.dump(f, js)
            
    def loadJSON(self, infilename):
        with open(infilename) as f:
            js = json.load(f)

        for name in js:
            setattr(self, name, js[name])
        
    def iterData(self):
        for j in range(self.numFolds):
            for k in range(len(self.featureSubsets)):
                yield j,k

    def taggedData(self, dataType, featureIndex, foldIndex):
        assert(dataType in self.taggedTypes)
        
        return self.taggedData[featureIndex][foldIndex][dataType]

    def untaggedData(self, dataType, featureIndex):
        assert(dataType in self.untaggedTypes)
        
        return self.untaggedData[featureIndex][dataType]

