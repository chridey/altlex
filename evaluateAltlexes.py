#read in JSON data set
#set aside 30% of data (~200 causal examples) for testing
#oversample remaining causal examples
#add features

import sys
import json
import collections
import itertools

import nltk

from featureExtractor import FeatureExtractor
from dataPoint import DataPoint
from randomForest import RandomForest
from naiveBayes import NaiveBayes
from adaBoost import AdaBoost
from svm import SVM
from gridSearchSVM import GridSearchSVM
from logisticRegression import LogisticRegression

classifierType = LogisticRegression #GridSearchSVM #RandomForest #AdaBoost #NaiveBayes #SVM #

with open(sys.argv[1]) as f:
    data = json.load(f)

if 0:
    count = collections.defaultdict(int)
    for dataPoint in data:
        dp = DataPoint(dataPoint)
        count[' '.join(dp.getAltlex())] += 1

    for s in sorted(count, key=count.get):
        print(s, count[s])
    exit()

#list of features
#curr stem ngrams
#prev stem ngrams
#altlex marker
#reporting verbs
#final reporting
#coref
#altlex length
#altlex pos ngrams
#cosine (maybe)

#first create dataset and assign features
fe = FeatureExtractor()
settingKeys = list(fe.experimentalSettings.keys())

for settingValues in itertools.product((True,False),
                                       repeat=len(settingKeys)):
    featureSettings = dict(zip(settingKeys, settingValues))
    featureSettings.update(fe.defaultSettings)
        
    print(featureSettings)

    causalSet = []
    nonCausalSet = []

    for dataPoint in data:
        #add pair of features dictionary and True or False
        dp = DataPoint(dataPoint)
    
        features = fe.addFeatures(dp, featureSettings)

    #if 'said' in lemmatizedAltlex or 'say' in lemmatizedAltlex or 'says' in lemmatizedAltlex:
    #    features['reporting'] = True

    #what about if the ONLY verb it contains is a reporting verb?
    #get list of verbs

    #what about length of altlex

    #features.update({"pronoun": "it" in altlex})

    #features.update({"firstsibling" + pos[len(altlex)]: True})

    #weirdly, these make things worse
    #features.update(getNgrams('pos1', dataPoint['sentences'][1]['pos']))
    #features.update(getNgrams('pos2', pos))

    #how about first verb of altlex
    #also add parts of speech for altlex
    #doesnt help

        if dataPoint['tag'] == 'causal':
            causalSet.append((features, True))
        else:
            nonCausalSet.append((features, False))

    print("True points: {} False points: {}".format(len(causalSet),
                                                    len(nonCausalSet)))

#now set aside at least 150 causal examples for testing or 30%, whichever is greater
    numCausalTesting = int(max(150, len(causalSet)*.3))
    numCausalTraining = len(causalSet) - numCausalTesting
    proportion = numCausalTesting/len(causalSet)
    numNonCausalTraining = int((1-proportion) * len(nonCausalSet))
    #oversamplingRatio = int(numNonCausalTraining/numCausalTraining)
    #numNonCausalTraining = numCausalTraining * oversamplingRatio
    #numNonCausalTesting = len(nonCausalSet)-numNonCausalTraining

#now set aside 30% for testing and oversample the causal training data to be balanced
    training = nonCausalSet[:numNonCausalTraining] + \
               causalSet[:numCausalTraining] #* oversamplingRatio
    testing = nonCausalSet[numNonCausalTraining:] + \
              causalSet[numCausalTraining:]

    #print(numCausalTesting, numCausalTraining, proportion, oversamplingRatio, numNonCausalTraining, numNonCausalTesting)
    #print(len(training), numNonCausalTraining/len(training),
    #      len(testing), numNonCausalTesting/len(testing))

    classifier = classifierType()

    if True:
        classifier.crossvalidate(training)
        classifier.show_most_informative_features(50)
        continue
    
    classifier.train(training)
    
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    truepos = 0
    trueneg = 0
    falsepos = 0
    falseneg = 0

    labels = set()
    for i, (feats, label) in enumerate(testing):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
        labels.add(label)
        if observed == label:
            if label == True:
                truepos += 1
            else:
                trueneg += 1
        elif label == False:
            falsepos +=1
        else:
            falseneg += 1
        
    for label in labels:
        print ('{} precision:'.format(label), nltk.metrics.precision(refsets[label], testsets[label]))
        print ('{} recall:'.format(label), nltk.metrics.recall(refsets[label], testsets[label]))
        print ('{} F-measure:'.format(label), nltk.metrics.f_measure(refsets[label], testsets[label]))

    print(truepos, trueneg, falsepos, falseneg)
    print(classifier.accuracy(testing))
    
    classifier.show_most_informative_features(50)

classifier.save("model")
