import json

from chnlp.utils.utils import splitData, sampleDataWithoutReplacement, iterFolds

def zipDataForCotraining(data):
    #data is currently
    #[[[feats00,label0], [feats01,label1], ...],
    #[[feats10,label0], [feats11,label1], ...]]
    #need to convert this into
    #[[[feats00,feats10],label0], [[feats01,feats11],label1], ...]

    if len(data) > 1:
        features1, labels1 = list(zip(*data[0]))
        features2, labels2 = list(zip(*data[1]))
        assert(labels1 == labels2)
        
        return list(zip(zip(features1,features2),labels1))
    
    return data[0]
    
def unzipDataForCotraining(data):
    features, labels = zip(*data)
    if type(features[0]) == dict:
        return data
    
    features1, features2 = zip(*features)
    return (list(zip(features1, labels)),
            list(zip(features2, labels)))

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

        self.taggedData = []
        self.untaggedData = []

        #first create the metadata and split the datasets for the tagged data
        tmpTaggedData = []
        tmpUntaggedData = []
        for i,featureSubset in enumerate(featureSubsets):
            setWithFeatures = makeDataset(taggedData,
                                          config.featureExtractor,
                                          config.featureSubset(*featureSubset))
            tmpTaggedData.append(setWithFeatures)
                
            #now randomly get U examples from the untagged data
            unknownSet = makeDataset(untaggedData,
                                     config.featureExtractor,
                                     config.featureSubset(*featureSubset),
                                     max=self.untaggedLimit)
            tmpUntaggedData.append(unknownSet)

        newTaggedData = zipDataForCotraining(tmpTaggedData)
        evaluation, testing = splitData(newTaggedData)

        #add these to some lookup table, blah blah
        for training, testing in iterFolds(evaluation,
                                           n_folds=self.numFolds):
            self.taggedData.append({'training': unzipDataForCotraining(training),
                                    'testing': unzipDataForCotraining(testing)})

        #newUntaggedData = zipDataForCotraining(tmpUntaggedData)
        newUntaggedData = tmpUntaggedData
        
        #sample here
        if self.untaggedSampleSize < len(newUntaggedData):
            startingData, remainingData = sampleDataWithoutReplacement(newUntaggedData,
                                                                       self.untaggedSampleSize)
        else:
            startingData = newUntaggedData
            remainingData = []

        for i in range(self.numFolds):
            self.untaggedData.append({'cotraining': startingData,
                                      'sampling': remainingData})

    def writeJSON(self, outfilename):
        #save num folds, feature subsets, tagged data, untagged data
        js = {'numFolds' : self.numFolds,
              'featureSubsets' : self.featureSubsets,
              'taggedData' : self.taggedData,
              'untaggedData' : self.untaggedData}

        with open(outfilename, 'w') as f:
            json.dump(js, f)
            
    def loadJSON(self, infilename):
        with open(infilename) as f:
            js = json.load(f)

        for name in js:
            setattr(self, name, js[name])
        
    def iterData(self):
        for j in range(self.numFolds):
            yield (self.taggedData[j]['training'],
                   self.taggedData[j]['testing'])

    def getTaggedData(self, dataType, foldIndex):
        assert(dataType in self.taggedTypes)
        
        return self.taggedData[foldIndex][dataType]

    def cotrainingData(self, foldIndex):
        return self.untaggedData[foldIndex]['cotraining']

    def samplingData(self, foldIndex):
        return self.untaggedData[foldIndex]['sampling']

    def updateTaggedData(self, newData, foldIndex):
        self.taggedData[foldIndex]['training'] = newData

    def updateUntaggedData(self,
                           foldIndex,
                           remainingUntaggedData,
                           remainingSample,
                           p,
                           n):

        numReplacements = len(self.featureSubsets) * (p+n)
        if len(remainingSample) > numReplacements:
            newUntaggedData, remainingSample = \
                             sampleDataWithoutReplacement(remainingSample,
                                                          numReplacements)
        else:
            newUntaggedData = remainingSample
            remainingSample = []
        
        #add these new points to the untagged data set
        untaggedData = list(remainingUntaggedData) + list(newUntaggedData)

        self.untaggedData[foldIndex] = {'cotraining': untaggedData,
                                        'sampling': remainingSample}
