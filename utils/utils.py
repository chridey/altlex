'''functions for manipulating data sets'''

import math
import random
import sys
import collections

if sys.version_info > (3,):
    from itertools import filterfalse
else:
    from itertools import ifilterfalse as filterfalse

from itertools import islice, chain

from sklearn.cross_validation import StratifiedKFold

def makeNgrams(words, max_ngrams, location=0):
    if location > 0:
        #split into first N, last N, and middle
        n = location
        ngrams = set()
        for length in range(1, max_ngrams+1):
            for i in range(max(0,len(words)-length+1-n),
                           len(words)-length+1):
                ngrams.add(('LAST',) + tuple(words[i:i+length]))
            for i in range(min(n,len(words)-length+1)):
                ngrams.add(('FIRST',) + tuple(words[i:i+length]))
            for i in range(min(n,len(words)-length+1),
                           max(0,len(words)-length+1-n)+1):
                ngrams.add(('MIDDLE',) + tuple(words[i:i+length]))
        return ngrams
    else:
        return set(chain(*(zip(*(words[i:] for i in range(j))) for j in range(1, max_ngrams+1))))

def indexedSubset(x, i):
    '''takes in a list x and a collection of indices i and returns
    the values in x that are at those indices'''

    #do what numpy arrays do but with any collection
    return list(zip(*filter(lambda x: x[0] in i, enumerate(x))))[1]

#sample data uniformly with replacement
def sampleDataWithReplacement(data, numSamples):
    sampleIndices = []
    for i in range(numSamples):
        sampleIndices.append(math.floor(random.random()*len(data)))

    return indexedSubset(data, sampleIndices)

def sampleDataWithoutReplacement(data, numSamples):
    remainingIndices = set(range(len(data)))
    sample = []
    for i in range(numSamples):
        sample.append(data[remainingIndices.pop()])

    return sample, indexedSubset(data, remainingIndices)

def splitData(data, proportion=.3, min=0):
    #now set aside at least 150 examples for testing or 30%, whichever is greater

    trueExamples = list(filter(lambda x:x[1], data))
    falseExamples = list(filterfalse(lambda x:x[1], data))

    numTrueTesting = int(max(min, len(trueExamples)*proportion))
    numTrueTraining = len(trueExamples) - numTrueTesting
    if len(trueExamples):
        proportion = 1.0*numTrueTesting/len(trueExamples)
    numFalseTraining = int((1-proportion) * len(falseExamples))
    #oversamplingRatio = int(numFalseTraining/numTrueTraining)
    #numFalseTraining = numTrueTraining * oversamplingRatio
    #numFalseTesting = len(falseExamples)-numFalseTraining

    #now set aside 30% for testing and oversample the true training data to be balanced
    print(len(trueExamples), len(falseExamples), numTrueTraining, numFalseTraining)
    training = falseExamples[:numFalseTraining] + \
               trueExamples[:numTrueTraining] #* oversamplingRatio
    testing = falseExamples[numFalseTraining:] + \
              trueExamples[numTrueTraining:]

    return training, testing

def balance(data, oversample=True):
    '''balancing for binary classification (labels must be T/F or 0/1)
    default is oversampling, if set to false, we subsample
    '''

    #first count the number in each class
    data = list(data)
    total = len(data)
    numTrue = sum(*islice(zip(*data),1,2))
    numFalse = total - numTrue

    #print(numTrue, numFalse)
    if numFalse > numTrue:
        if oversample:
            oversamplingRatio = int(numFalse/numTrue)
            numFalse = numTrue * oversamplingRatio
            return list(filter(lambda x:x[1], data)) * oversamplingRatio + \
                   list(filterfalse(lambda x:x[1], data))[:numFalse]
        else:
            return list(filter(lambda x:x[1], data)) + \
                   list(filterfalse(lambda x:x[1], data))[:numTrue]

    return data

def balance(data, oversample=True, seed=None, bootstrap=False, verbose=False):
    data = list(data)
    total = len(data)
    
    balancedData = collections.defaultdict(list)
    for datum in data:
        balancedData[datum[1]].append(datum)

    if verbose:
        for label in balancedData:
            print(len(balancedData[label]))
            
    if type(oversample) == int:
        num = oversample
    elif oversample:
        num = max(len(balancedData[i]) for i in balancedData)
    else:
        num = min(len(balancedData[i]) for i in balancedData)

    random.seed()
    finalData = []
    for label in balancedData:
        if len(balancedData[label]) == num:
            finalData.extend(balancedData[label])
        else:
            for i in range(num):
                if bootstrap:
                    index = i % len(balancedData[label])
                else:
                    index = int(random.random()*len(balancedData[label]))
                finalData.append(balancedData[label][index])
    return finalData
    
def iterFolds(data, n_folds=2, random_state=1):
    features, y = zip(*data)
    skf = StratifiedKFold(y,
                          n_folds=n_folds,
                          random_state=random_state)

    for train_index,test_index in skf:
        train = indexedSubset(data, set(train_index))
        balancedData = balance(train)

        test = indexedSubset(data, set(test_index))

        yield balancedData, test
