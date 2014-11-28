#read in JSON data set
#set aside 30% of data (~200 causal examples) for testing
#oversample remaining causal examples
#add features

import sys
import json
import collections

import nltk

from featureExtractor import FeatureExtractor
from dataPoint import DataPoint

with open(sys.argv[1]) as f:
    data = json.load(f)

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
causalSet = []
nonCausalSet = []
fe = FeatureExtractor()
featureSettings = fe.defaultSettings()

for dataPoint in data:
    #add pair of features dictionary and True or False
    dp = DataPoint(dataPoint)
    
    features = fe.addFeatures(dp, featureSettings)

    #if 'said' in lemmatizedAltlex or 'say' in lemmatizedAltlex or 'says' in lemmatizedAltlex:
    #    features['reporting'] = True

    #features.update({"cosine" : cs.scoreWords(prevStems,
    #                                          currStems[len(altlex):]) > .05 })
    features.update({"coref": "this" in altlex or "that" in altlex})


    #what about if the ONLY verb it contains is a reporting verb?
    #get list of verbs

    #what about length of altlex

    features.update({"pronoun": "it" in altlex})

    #features.update({"firstsibling" + pos[len(altlex)]: True})

    #intersecting words
    #this really helps with precision
    features['intersection'] = len(set(currStems) & set(prevStems))

    #weirdly, these make things worse
    #features.update(getNgrams('pos1', dataPoint['sentences'][1]['pos']))
    #features.update(getNgrams('pos2', pos))

    if len(altlex):
        features.update(getNgrams('altlex pos', pos[:len(altlex)]))

        #how about first verb of altlex
        #also add parts of speech for altlex
        #doesnt help

        #get all the verbs in the sentence and determine overlap with reporting verbs
        #or maybe just the last verb?
        
        verbs = []
        for (index,p) in enumerate(pos[:len(altlex)]):
            if p[0] == 'V':
                #features['first altlex verb: ' + currStems[index]] = True
                verbs.append(currLemmas[index])
                break

        try:
            if len(set(wn.synsets(verbs[-1], pos=wn.VERB)) & reporting) > 0:
                features['reporting'] = True
        except IndexError:
            print (altlex, dataPoint['sentences'][0]['parse'])
        
        #if len(set(wn.synsets(lemmatizedAltlex[-1], pos=wn.VERB)) & reporting) > 0:
        #    features['reporting'] = True
        #first verb part of speech of altlex?

        #what about "AltLex contains an explicit discourse marker"?            
        alower = ' '.join(altlex)
        for marker in markers:
            if marker in altlex or len(marker.split()) > 1 and marker in alower:
                features['marker ' + marker] = True
                
    #for marker in markers:
    #    if marker in currStems[len(altlex):]:
    #        features['sentence marker ' + marker] = True

    features.update(getNgrams('stem1', prevStems))
    features.update(getNgrams('stem2', currStems))
        
    if dataPoint['tag'] == 'causal':
        causalSet.append((features, True))
    else:
        nonCausalSet.append((features, False))

#now set aside at least 200 causal examples for testing or 30%, whichever is greater
numCausalTesting = int(max(200, len(causalSet)*.3))
numCausalTraining = len(causalSet) - numCausalTesting
proportion = numCausalTesting/len(causalSet)
numNonCausalTraining = int((1-proportion) * len(nonCausalSet))
oversamplingRatio = int(numNonCausalTraining/numCausalTraining)
numNonCausalTraining = numCausalTraining * oversamplingRatio
numNonCausalTesting = len(nonCausalSet)-numNonCausalTraining

print(numCausalTesting, numCausalTraining, proportion, oversamplingRatio, numNonCausalTraining, numNonCausalTesting)

#now set aside 30% for testing and oversample the causal training data to be balanced
training = nonCausalSet[:numNonCausalTraining] + \
           causalSet[:numCausalTraining] * oversamplingRatio
testing = nonCausalSet[numNonCausalTraining:] + \
          causalSet[numCausalTraining:]

print(len(training), numNonCausalTraining/len(training),
      len(testing), numNonCausalTesting/len(testing))

classifier = nltk.NaiveBayesClassifier.train(training)

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
print(nltk.classify.accuracy(classifier, testing))

classifier.show_most_informative_features(50)
