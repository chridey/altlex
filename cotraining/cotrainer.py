import itertools
import collections

from chnlp.ml.sklearner import Sklearner
from chnlp.utils.utils import indexedSubset

class NotProbabilistic(Exception):
    pass

class Cotrainer(Sklearner):
    def __init__(self, classifier1, classifier2, **kwargs):
        self.classifiers = [classifier1, classifier2]
        super().__init__(**kwargs)

    #get the probabilities for all the data points and take the
    #p most confident positive points and n most confident negative points
    def cotrain(self, classifierIndex, p, n, taggedData, untaggedData):
        self.classifiers[classifierIndex].train(taggedData)

        queue = collections.defaultdict(list)
        for i,datum in enumerate(untaggedData):
            #datum is tuple of features,label
            probs = self.classifiers[classifierIndex].prob(datum[0])
            for k in range(len(probs)):
                queue[k].append((probs[k], i))

        topNegative = sorted(queue[0], reverse=True)[:n]
        topPositive = sorted(queue[1], reverse=True)[:p]

        print(topNegative, topPositive)
        
        #now get the indices for each of these
        negativeIndices = set(*itertools.islice(zip(*topNegative),1,2))
        positiveIndices = set(*itertools.islice(zip(*topPositive),1,2))

        newTaggedData = list(indexedSubset(untaggedData, negativeIndices))
        newPositives = list(zip(*indexedSubset(untaggedData, positiveIndices)))[0]
        newTaggedData += list(zip(newPositives,
                                  [True] * p))

        remainingIndices = set(range(len(untaggedData))) - negativeIndices - positiveIndices
        remainingUntaggedData = indexedSubset(untaggedData, remainingIndices)
            
        return newTaggedData, remainingUntaggedData

    def accuracy(self, testing, transform=True):
        raise NotImplementedError

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

    def metrics(self, testing, transform=True):
        #testing is currently
        #[[[feats00,label0], [feats01,label1], ...],
        #[[feats10,label0], [feats11,label1], ...],
        #...]
        #need to convert this into
        #[[[feats00,feats10,...],label0], [[feats01,feats11,...],label1], ...]

        features1 = list(*itertools.islice(zip(*testing[0]),0,1))
        features2 = list(*itertools.islice(zip(*testing[1]),0,1))
        labels1 = list(*itertools.islice(zip(*testing[0]),1,2))
        labels2 = list(*itertools.islice(zip(*testing[1]),1,2))
        assert(labels1 == labels2)
        
        newTesting = list(zip(zip(features1,features2),labels1))
        return super().metrics(newTesting, transform)
