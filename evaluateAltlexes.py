#read in JSON data set
#set aside 30% of data (~200 causal examples) for testing
#oversample remaining causal examples
#add features

import sys
import json
import collections
import itertools

import nltk

from utils import splitData
from featureExtractor import FeatureExtractor
from dataPoint import DataPoint
from randomForest import RandomForest
from naiveBayes import NaiveBayes
from adaBoost import AdaBoost
from svm import SVM
from gridSearch import GridSearchSVM, GridSearchLogit
from logisticRegression import LogisticRegression

classifierType = LogisticRegression #RandomForest #GridSearchLogit #AdaBoost #SVM #GridSearchSVM #NaiveBayes #

with open(sys.argv[1]) as f:
    data = json.load(f)

#first create dataset and assign features
fe = FeatureExtractor()
settingKeys = list(fe.experimentalSettings)

for settingValues in itertools.product((True,False),
                                       repeat=len(settingKeys)):
    featureSettings = fe.defaultSettings.copy()
    featureSettings.update(dict(zip(settingKeys, settingValues)))
        
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

    training,testing = splitData(causalSet, nonCausalSet)
    #print(numCausalTesting, numCausalTraining, proportion, oversamplingRatio, numNonCausalTraining, numNonCausalTesting)
    #print(len(training), numNonCausalTraining/len(training),
    #      len(testing), numNonCausalTesting/len(testing))

    classifier = classifierType()

    if True:
        classifier.crossvalidate(training)
    else:
        classifier.train(training)

    classifier.show_most_informative_features(50)
    continue

    accuracy = classifier.accuracy(testing)
    precision, recall = classifier.metrics(testing)
    classifier.printResults(accuracy, precision, recall)

classifier.save("model")
