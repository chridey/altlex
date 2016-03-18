#after each round
#classify new examples
#update labels
#get features for new data
#maybe recalculate KLD
#train on new labels

import operator
import argparse
import json
import collections
import random

import numpy as np

from sklearn.externals import joblib

from altlex.misc import extractFeatures
from altlex.misc import trainFeatureWeights
from altlex.evaluation.annotations.causalAnnotations import getCausalAnnotations
from altlex.utils.readers.bootstrapAlignedParsedPairIterator import BootstrapAlignedParsedPairIterator

from altlex.ml.gridSearch import HeldOutGridSearch
from altlex.ml import sklearner

from altlex.featureExtraction import featureExtractor
from altlex.featureExtraction.dataPointMetadata import DataPointMetadataList 

from altlex.utils import wordUtils

#only include them if 1) the predicted class matches a possible altlex for that class
#and 2) the predictions for the pair are the same
def filterMatchedAltlexes(dataset, knownAltlexes, predictions, scores, minCount=1):

    idLookup = {}
    for i in range(0,len(dataset),2):
        assert(dataset[i].datumId == dataset[i+1].datumId)
        
        if predictions[i] != predictions[i+1]:
            continue
        prediction = predictions[i]
        
        if tuple(dataset[i].altlex) in knownAltlexes[prediction] or tuple(dataset[i+1].altlex) in knownAltlexes[prediction]:
            harmonicMean = abs((2*scores[i]*scores[i+1])/(scores[i]+scores[i+1]))
            idLookup[dataset[i].datumId] = harmonicMean,prediction

    return idLookup

def updateLabels(dataset, idLookup, classes, indices=None, verbose=False, n='all', balance=True):

    classCounts = collections.Counter(j[1] for j in idLookup.values())
    smallestClass,smallestCounts = min(classCounts.items(), key=lambda x:x[1])

    if verbose:
        print(classCounts)
        print(smallestClass,smallestCounts)

    if balance == True:
        balanceFactor = [1]*len(classes)
    else:
        balanceFactor = [None]*len(classes)
        for i in classes:
            if classCounts[i] == smallestCounts:
                balanceFactor[i] = 1
            else:
                balanceFactor[i] = 1.*classCounts[i] / smallestCounts

    if verbose:
        print(balanceFactor)

    #if n is 'all', just take all of the smallest class and subsample the rest
    labelMap = {}
    if n == 'all':
        for i in classes:
            if classCounts[i] == smallestCounts:
                labelMap.update({j:k[1] for j,k in idLookup.items() if k[1] == i})
            else:
                currentClassIds = dict(filter(lambda x:x[1][1]==i, idLookup.items()))
                numSampled = int(smallestCounts*balanceFactor[i])

                if verbose:
                    print(i,numSampled)
                
                for datumId in np.random.choice(list(currentClassIds.keys()), numSampled, False):
                    labelMap[datumId] = i
    else:
        #otherwise sort the points by the harmonic mean of their confidences and
        #take the n most confident points in each class
        for i in classes:
            #sort in descending order
            currentClassIds = sorted(filter(lambda x:x[1][1]==i,
                                            idLookup.items()),
                                     key=lambda x:x[1][0],
                                     reverse=True)
            labelMap.update({j:k[1] for j,k in currentClassIds[:int(balanceFactor[i]*n*2)]})

    print(len(labelMap))

    newdataset = []
    for i in dataset:
        if i.datumId in labelMap:
            i.label = labelMap[i.datumId]
            newdataset.append(i)

    return DataPointMetadataList(newdataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='bootstrap and evaluate a classifier on a dataset with altlexes')
    parser.add_argument('trainfile', 
                        help='the training metadata in gzipped JSON format')
    parser.add_argument('parsedir', 
                        help='the paired pairses in gzipped JSON format')
    parser.add_argument('alignedlabels', 
                        help='the aligned labeled parses in gzipped JSON format')
    parser.add_argument('--testdir',
                        help='annotated test directory to extract altlexes from')

    parser.add_argument('--start_iteration',
                        type=int,
                        default=0)
    parser.add_argument('--iterations',
                        type=int,
                        default=10)

    parser.add_argument('--search_parameters',
                        type=json.loads,
                        default={'penalty': ['l2'],
                                 'alpha': [0.0001]})
    parser.add_argument('--batch_size',
                        type = int,
                        default = 100)
    parser.add_argument('--num_epochs',
                        type = int,
                        default = 50)

    parser.add_argument('--config',
                        help='config file for feature extractor in JSON format')

    parser.add_argument('--filter')
    parser.add_argument('--ablate')    
    parser.add_argument('--interaction',
                        type=json.loads)

    parser.add_argument('--balance',
                        action='store_true')
    parser.add_argument('--combined',
                        action='store_true')

    parser.add_argument('-v', '--verbose',
                        action='store_true')

    parser.add_argument('--load')
    parser.add_argument('--save')

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            settings = json.load(f)
        fe = featureExtractor.FeatureExtractor(settings, verbose=True)
    else:
        fe = featureExtractor.FeatureExtractor(verbose=True)
    
    train = DataPointMetadataList.load(args.trainfile)
    train = featureExtractor.createModifiedDataset(train,
                             args.filter,
                             args.ablate,
                             args.interaction)

    labelLookup = wordUtils.trinaryCausalSettings[1]
    print(len(train))

    parameters = dict(n_iter=args.num_epochs,
                      shuffle=True,
                      batch_size=args.batch_size,
                      verbose=False)

    test = getCausalAnnotations(args.testdir,
                                train.causalAltlexes,
                                fe,
                                labelLookup).dedupe(train, True)

    test = featureExtractor.createModifiedDataset(test,
                                                  args.filter,
                                                  args.ablate,
                                                  args.interaction)
    

    if args.load:
        classifier = joblib.load(args.load)
        dict_vectorizer = joblib.load(args.load + '.vectorizer')

        trainFeatureWeights.evaluate(test, classifier, dict_vectorizer,
                                     combined=args.combined)
    else:
        test = getCausalAnnotations(args.testdir,
                                    train.causalAltlexes,
                                    fe,
                                    labelLookup).dedupe(train, True)

        test = featureExtractor.createModifiedDataset(test,
                                args.filter,
                                args.ablate,
                                args.interaction)
        tune = test
        
        classifier = HeldOutGridSearch('mini_batch_sgd',
                                       parameters,
                                       args.search_parameters,
                                       verbose=args.verbose,
                                       transformer=sklearner.Identity())

        classifier,dict_vectorizer = trainFeatureWeights.main(train, test, classifier,
                                                              tune, balance=args.balance,
                                                              combined=args.combined)

    prevTrain = train
    for iteration in range(args.iterations):
        #first identify possible new datapoints, using the known altlexes from the previous iteration
        knownAltlexes = set()
        for i in prevTrain.altlexes:
            knownAltlexes.update(prevTrain.altlexes[i].keys())
        iterator = BootstrapAlignedParsedPairIterator(args.parsedir, None,
                                                      knownAltlexes, prevTrain.datumIndices,
                                                      verbose=True)
        iterator.load(args.alignedlabels)

        print(args.combined)
        if not args.combined:
            knownAltlexes = prevTrain.altlexes
        else:
            knownAltlexes = prevTrain.combinedAltlexes
        print(knownAltlexes.keys())
        print([len(knownAltlexes[i]) for i in knownAltlexes])

        dataset = extractFeatures.main(iterator, fe)
        dataset = featureExtractor.createModifiedDataset(dataset,
                                                         args.filter,
                                                         args.ablate,
                                                         args.interaction)

        print(len(dataset))

        new_features = dict_vectorizer.transform(data.features for data in dataset)
        print(new_features.shape)
        predictions = classifier.predict(new_features)
        scores = classifier.decision_function(new_features)
        print(sum(predictions))
        print(set(predictions))
                    
        idLookup = filterMatchedAltlexes(dataset, knownAltlexes, predictions, scores)
        print(len(idLookup))
        train = updateLabels(dataset, idLookup, knownAltlexes.keys(), verbose=True)
        print(len(train))
        
        altlexes = train.altlexes
        for i in altlexes:
            print(i, len(altlexes[i]), sum(altlexes[i].values()))
            print('*'*79)
            for j in sorted(altlexes[i].items(), key=lambda x:x[1])[-30:]:
                print(j)
            print('*'*79)

        train = DataPointMetadataList(train + prevTrain)
        
        test = getCausalAnnotations(args.testdir,
                                    train.causalAltlexes,
                                    fe,
                                    labelLookup).dedupe(train, True)
        test = featureExtractor.createModifiedDataset(test,
                                args.filter,
                                args.ablate,
                                args.interaction)

        tune = test
        classifier = HeldOutGridSearch('mini_batch_sgd',
                                       parameters,
                                       args.search_parameters,
                                       verbose=args.verbose,
                                       transformer=sklearner.Identity())
        classifier,dict_vectorizer = trainFeatureWeights.main(train, classifier, test,
                                                              tune, balance=args.balance,
                                                              combined=args.combined)
        prevTrain = train
        
        if args.save:
            joblib.dump(classifier.classifier, '{}.{}'.format(args.save, iteration))
            joblib.dump(dict_vectorizer, '{}.{}.vectorizer'.format(args.save, iteration))
            train.save('{}.{}.train'.format(args.save, iteration))
            test.save('{}.{}.test'.format(args.save, iteration))
