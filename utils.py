
def indexedSubset(x, i):
    '''takes in a list x and a collection of indices i and returns
    the values in x that are at those indices'''
    
    return list(zip(*filter(lambda x: x[0] in i, enumerate(x))))[1]

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
