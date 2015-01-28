import itertools
import collections

from chnlp.ml.sklearner import Sklearner
from chnlp.utils.utils import indexedSubset
from chnlp.cotraining.cotrainingDataHandler import unzipDataForCotraining

class NotProbabilistic(Exception):
    pass

class Cotrainer(Sklearner):
    def __init__(self, classifierType, **kwargs):
        self.classifiers = [classifierType(), classifierType()]
        super().__init__(**kwargs)

    #get the probabilities for all the data points and for each classifier  take the
    #p most confident positive points and n most confident negative points
    def train(self, taggedData, untaggedData, p, n):
        for index in range(len(self.classifiers)):
            unzippedTaggedData = taggedData #unzipDataForCotraining(taggedData)
            unzippedUntaggedData = untaggedData #unzipDataForCotraining(untaggedData)

            self.classifiers[index].train(unzippedTaggedData[index])

            queue = collections.defaultdict(list)
            for i,datum in enumerate(unzippedUntaggedData[index]):
                #datum is tuple of features,label
                probs = self.classifiers[index].prob(datum[0])
                for k in range(len(probs)):
                    queue[k].append((probs[k], i))

            topNegative = sorted(queue[0], reverse=True)[:n]
            topPositive = sorted(queue[1], reverse=True)[:p]

            #print(topNegative, topPositive)
        
            #now get the indices for each of these
            negativeIndices = set(*itertools.islice(zip(*topNegative),1,2))
            positiveIndices = set(*itertools.islice(zip(*topPositive),1,2))

            #print(negativeIndices, positiveIndices)
            newNegatives = list(zip(*indexedSubset(untaggedData, negativeIndices)))[0]
            taggedData += list(zip(newNegatives, [False] * n))
            newPositives = list(zip(*indexedSubset(untaggedData, positiveIndices)))[0]
            taggedData += list(zip(newPositives,
                                  [True] * p))

            remainingIndices = set(range(len(untaggedData))) - negativeIndices - positiveIndices
            untaggedData = indexedSubset(untaggedData, remainingIndices)
            
        return taggedData, untaggedData

    def accuracy(self, testing, transform=True):
        #TODO
        return 0

    def metrics(self, testing, transform=True):
        super().metrics(zipDataForCotraining(testing), transform)

    def classify(self, features, transform=True):
        #need to get the probability of each class for
        #each classifier and then determine which class prob is greatest

        assert(type(features) in (tuple,list))

        probs = [1 for i in self.classifiers[0].numClasses]
        for i,classifier in enumerate(self.classifiers):
            try:
                prob = classifier.prob(features[i], transform)
            except NotImplementedError:
                raise NotProbabilistic('The classifiers for co-training must be probabilistic')

            for j in range(len(probs)):
                probs[j] *= prob[j]
            
        return probs.index(max(probs))

class SelfTrainer(Cotrainer):
    def __init__(self, classifierType, classifierParams=None, **kwargs):
        if classifierParams:
            assert(type(classifierParams) == dict)
            self.classifiers = [classifierType(**classifierParams)]
        else:
            self.classifiers = [classifierType()]

        super(Cotrainer, self).__init__(**kwargs)

    def metrics(self, testing, transform=True):
        super(Cotrainer, self).metrics(testing[0], transform)
        
    def classify(self, features, transform=True):
        return self.classifiers[0].classify(features, transform)
                         
    '''
    def metrics(self, testing, transform=True):
        #testing is currently
        #[[[feats00,label0], [feats01,label1], ...],
        #[[feats10,label0], [feats11,label1], ...],
        #...]
        #need to convert this into
        #[[[feats00,feats10],label0], [[feats01,feats11],label1], ...]

        features1 = list(*itertools.islice(zip(*testing[0]),0,1))
        features2 = list(*itertools.islice(zip(*testing[1]),0,1))
        labels1 = list(*itertools.islice(zip(*testing[0]),1,2))
        labels2 = list(*itertools.islice(zip(*testing[1]),1,2))
        assert(labels1 == labels2)
        
        newTesting = list(zip(zip(features1,features2),labels1))
        return super().metrics(newTesting, transform)
    '''
