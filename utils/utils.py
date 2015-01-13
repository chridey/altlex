import math
import random

def indexedSubset(x, i):
    '''takes in a list x and a collection of indices i and returns
    the values in x that are at those indices'''
    
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

def splitData(trueExamples, falseExamples, proportion=.3, min=150):
    #now set aside at least 150 causal examples for testing or 30%, whichever is greater
    numTrueTesting = int(max(min, len(trueExamples)*proportion))
    numTrueTraining = len(trueExamples) - numTrueTesting
    proportion = numTrueTesting/len(trueExamples)
    numFalseTraining = int((1-proportion) * len(falseExamples))
    #oversamplingRatio = int(numFalseTraining/numTrueTraining)
    #numFalseTraining = numTrueTraining * oversamplingRatio
    #numFalseTesting = len(falseExamples)-numFalseTraining

#now set aside 30% for testing and oversample the true training data to be balanced
    training = falseExamples[:numFalseTraining] + \
               trueExamples[:numTrueTraining] #* oversamplingRatio
    testing = falseExamples[numFalseTraining:] + \
              trueExamples[numTrueTraining:]

    return training, testing
