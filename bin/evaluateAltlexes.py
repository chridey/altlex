#read in JSON data set
#set aside 30% of data (~200 causal examples) for testing
#oversample remaining causal examples
#add features

import json
import argparse
import itertools
import time
import gzip

from chnlp.utils.utils import splitData,balance
from chnlp.altlex.featureExtractor import makeDataset
from chnlp.altlex.config import Config

from chnlp.altlex.dataPoint import DataPoint

if __name__ == '__main__':
    config = Config()

    parser = argparse.ArgumentParser(description='train and/or evaluate a classifier on a dataset with altlexes')

    parser.add_argument('infile', 
                        help='the file containing the sentences and metadata in JSON format')
    parser.add_argument('--classifier', '-c', metavar='C',
                        choices = config.classifiers.keys(),
                        default = config.classifier,
                        help = 'the supervised learner to use (default: %(default)s) (choices: %(choices)s)')

    parser.add_argument('--crossvalidate', '-v', action='store_true',
                        help = 'crossvalidate and print the results (default: train only)')
    parser.add_argument('--numFolds', '-k', type=int, default=2,
                        help='the number of folds for crossvalidation (default: %(default)s)')

    parser.add_argument('--unlabeled', '-u', metavar='U',
                        help = 'use additional unlabeled data (requires that the classifier be an unsupervised or semi-supervised learning algorithm')

    parser.add_argument('--test', '-t', action='store_true',
                        help = 'test on the set aside data (default: train only)')
    parser.add_argument('--testfile', 
                        help = 'test on the set aside data (default: train only)')

    parser.add_argument('--all', '-a', action='store_true',
                        help = 'use all data for training or cross-validation (default:set aside some for testing)')

    parser.add_argument('--balance', '-b', type=int, default=1, choices = (0,1,2),
                        help = 'whether to balance the data by 0) not at all, 1) oversampling the positive class, or 2) undersampling the negative class (default: %(default)s)')

    parser.add_argument('--save', metavar = 'S',
                        help = 'save the model to a file named S')

    parser.add_argument('--printErrorAnalysis', '-p', action='store_true',
                        help = 'save the model to a file named S')
    parser.add_argument('--analyze_features', action='store_true',
                        help = 'analyze feature weights')

    parser.add_argument('--gz', action='store_true')

    parser.add_argument('--config',
                        help = 'JSON config file to use instead of command line options')

    parser.add_argument('--features',
                        help = 'comma-separated list of features, overrides --config option')

    parser.add_argument('--maxSupervised', type=float, default=float('inf'))

    args = parser.parse_args()

    #start with the config file options, but allow them to be overwritten by command line
    if args.config:
        with open(args.config) as f:
            config.setParams(json.load(f))

    if args.features:
        features = set(args.features.split(','))
        config.setFeatures(features)

    classifierType = config.classifiers[args.classifier]

    print('loading data...')
    starttime = time.time()
    if args.gz:
        with gzip.open(args.infile, 'rb') as f:
            data = json.load(f)
    else:
        with open(args.infile) as f:
            data = json.load(f)
    if args.testfile:
        with open(args.testfile) as f:
            testdata = json.load(f)
    else:
        testdata = None
    print(len(data))
    print(len(testdata) if testdata is not None else 0)
    print(time.time()-starttime)
    
    if args.unlabeled:
        assert(args.classifier in config.semisupervised or args.classifier in config.unsupervised)

        with open(args.unlabeled) as f:
            untaggedData = json.load(f)
    else:
        untaggedData = []

    #first create dataset and assign features
    settingKeys = list(config.experimentalSettings)
    print(settingKeys)
    #for settingValues in itertools.product((True,False),
    #                                       repeat=len(settingKeys)):
    for settingKey in settingKeys + ['']:
        featureSettings = config.fixedSettings.copy()
        #    featureSettings.update(dict(zip(settingKeys, settingValues)))
        if settingKey != '':
            featureSettings[settingKey] = True

        for i in featureSettings:
            if featureSettings[i]:
                print(i)

        '''
        #CHANGE!!!!
        newData = data
        data = []
        for d in newData:
            if d['altlexLength'] == 0:
                data.append(d)
        '''

        starttime = time.time()
        print('extracting features...')
        taggedSet = makeDataset(data,
                                config.featureExtractor,
                                featureSettings,
                                max=args.maxSupervised)
        print(time.time()-starttime)
        
        if args.all:
            training,testing = taggedSet,[]
        elif testdata:
            training = taggedSet
            testing = makeDataset(testdata,
                                  config.featureExtractor,
                                  featureSettings)
        else:
            training,testing = splitData(taggedSet)

        if len(untaggedData):
            untaggedSet = makeDataset(untaggedData,
                                      config.featureExtractor,
                                      featureSettings)
        else:
            untaggedSet = []

        #print(len(training), numNonCausalTraining/len(training),
        #      len(testing), numNonCausalTesting/len(testing))

        classifier = classifierType()

        from itertools import islice
        total = len(training)
        numTrue = sum(*islice(zip(*training),1,2))
        numFalse = total - numTrue

        print(numTrue, numFalse)
        total = len(testing)
        if total>0:
            numTrue = sum(*islice(zip(*testing),1,2))
            numFalse = total - numTrue

            print(numTrue, numFalse)


        print("done making data")
        if args.crossvalidate:
            classifier.crossvalidate(training,
                                     n_folds=args.numFolds,
                                     training=untaggedSet,
                                     balanced=args.balance,
                                     printErrorAnalysis=args.printErrorAnalysis)

        else:
            if args.balance:
                training = balance(training,
                                   oversample=args.balance==1)
            total = len(training)
            numTrue = sum(*islice(zip(*training),1,2))
            numFalse = total - numTrue

            print(numTrue, numFalse)
            classifier.train(training + untaggedSet)

        if args.analyze_features:
            classifier.show_most_informative_features(250)

        if args.test or args.testfile:
            X,y = classifier.transform(testing)
            accuracy = classifier.accuracy(X, labels=y, transform=False)
            precision, recall = classifier.metrics(X, labels=y, transform=False)
            classifier.printResults(accuracy, precision, recall)

            '''
            #for each point in testing, shorten datapoint.data
            for t in testing:
                origLength = t.data['altlexLength']
                maxScore = 0
                maxScoreClass = None
                for i in range(len(t.data['sentences'][0]['words'])):
                    t.data['altlexLength'] = i+1
                    d = makeDataset([t.data],
                                    config.featureExtractor,
                                    featureSettings)
                    #print(DataPoint(d[0].data).getAltlex())
                    #print(d[0])
                    score = classifier.confidence(d[0][0])
                    if abs(score) > maxScore:
                        maxScore = abs(score)
                        maxScoreClass = classifier.classify(d[0][0])
                        maxScoreAltLex = DataPoint(d[0].data).getAltlex()
                print(maxScore)
                print(maxScoreClass)
                print(maxScoreAltLex)
                t.data['altlexLength'] = origLength
                print(DataPoint(t.data).getAltlex())
                print(DataPoint(t.data).getCurrWords())
                print(t.data['tag'])
                print()
            '''

        classifier.close()

        print(config.featureExtractor.pos)
    if args.save:
        classifier.save(args.save)
