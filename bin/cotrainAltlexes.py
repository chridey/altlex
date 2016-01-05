#train a classifier on semantic features
#train a classifier on all the other features

#use each of those classifiers to label the random data from the untagged data

#add these examples to the training set
#replenish the untagged data

import json
import argparse
import collections
import itertools

from chnlp.metrics.metricsAccumulator import MetricsAccumulator

from chnlp.altlex.featureExtractor import makeDataset
from chnlp.altlex.config import Config

from chnlp.cotraining.cotrainer import Cotrainer, SelfTrainer, NotProbabilistic
from chnlp.cotraining.cotrainingDataHandler import CotrainingDataHandler, unzipDataForCotraining

config = Config()

parser = argparse.ArgumentParser(description='train and/or evaluate a classifier on a dataset with altlexes')

parser.add_argument('taggedFile', 
                    help='the file with tags containing the sentences and metadata in JSON format')
parser.add_argument('untaggedFile', 
                    help='the file without tags containing the sentences and metadata in JSON format')

parser.add_argument('--positive', '-p', metavar='P', type=int, required=True,
                    help='the number of positive examples to sample from U_prime')

parser.add_argument('--negative', '-n', metavar='N', type=int, required=True,
                    help='the number of negative examples to sample from U_prime')

parser.add_argument('--unlabeled', '-u', metavar='U', type=float, default=float('inf'),
                    help='the number of unlabeled examples to sample at random (default: %(default)s) (use all the data) (P + N must be less than U)')

parser.add_argument('--limit', '-l', metavar='L', type=float, default=float('inf'),
                    help='the number of samples to keep in reserve (default: %(default)s)')

parser.add_argument('--iterations', '-k', metavar='K', type=float, default=float('inf'),
                    help='the number of iterations (default: %(default)s) (iterate until all data is tagged)')

parser.add_argument('--numFolds', type=int, default=2,
                    help='the number of folds for crossvalidation (default: %(default)s)')

parser.add_argument('--classifier', '-c', metavar='C',
                    choices = config.classifiers.keys(),
                    default = config.classifier,
                    help = 'the supervised learner to use (default: %(default)s) (choices: %(choices)s)')

parser.add_argument('--selftrain', '-s', action='store_true',
                    help='do self-training instead of cotraining')                    

parser.add_argument('--dump', metavar = 'D',
                    help = 'dump the processed features to a file named D')

parser.add_argument('--load', metavar = 'L',
                    help = 'load processed features from a file named L')

parser.add_argument('--save', metavar = 'S',
                    help = 'save the model to a file named S')

parser.add_argument('--config',
                    help = 'JSON config file to use instead of command line options')

args = parser.parse_args()

#start with the config file options, but allow them to be overwritten by command line
if args.config:
    with open(args.config) as f:
        config.setParams(json.load(f))

handler = CotrainingDataHandler(args.limit,
                                args.unlabeled)

featureSubsets = (
    #('semantic',),
    ('syntactic', 'lexical', 'structural'),
    ('semantic',),
    )
if args.selftrain:
    featureSubsets = tuple(itertools.chain(*featureSubsets)),
    cotrainer = SelfTrainer(config.classifiers[args.classifier])
    metrics = MetricsAccumulator(args.numFolds,
                                 featureSubsets)
else:
    cotrainer = Cotrainer(config.classifiers[args.classifier])
    metrics = MetricsAccumulator(args.numFolds,
                                 featureSubsets + ('combined',))

if args.load:
    handler.loadJSON(args.load)
else:
    with open(args.taggedFile) as f:
        taggedData = json.load(f)

    with open(args.untaggedFile) as f:
        untaggedData = json.load(f)

    handler.makeDataset(taggedData,
                        untaggedData,
                        args.numFolds,
                        featureSubsets,
                        makeDataset,
                        config)

if args.dump:
    handler.writeJSON(args.dump)

try:
    i = 0
    while i < args.iterations:
        
        print('iteration {}'.format(i))
        
        for foldIndex, (training, testing) in enumerate(handler.iterData()):
            untaggedData = handler.cotrainingData(foldIndex)
            remainingSample = handler.samplingData(foldIndex)
        
            print('training is {}'.format(len(training[0])))
            print('testing is {}'.format(len(testing[0])))
            print('untagged is {}'.format(len(untaggedData[0])))
            print('remaining sample is {}'.format(len(remainingSample)))
                      
            newTaggedData, remainingUntaggedData = cotrainer.train(training,
                                                                   untaggedData,
                                                                   args.positive,
                                                                   args.negative)
            #print(len(newTaggedData))
            #print(len(remainingUntaggedData))
            
            #add new tagged data to training
            handler.updateTaggedData(newTaggedData, foldIndex)

            #replenish the points that were removed
            handler.updateUntaggedData(foldIndex,
                                       remainingUntaggedData,
                                       remainingSample,
                                       args.positive,
                                       args.negative)

            
            #newTesting = unzipDataForCotraining(testing)
            for featureIndex in range(len(featureSubsets)):
                
                metrics.add(foldIndex,
                            featureSubsets[featureIndex],
                            cotrainer.classifiers[featureIndex],
                            #newTesting[featureIndex])
                            testing[featureIndex])

            if 'combined' in metrics.graphNames:
                try:
                    metrics.add(foldIndex,
                                'combined',
                                cotrainer,
                                testing)
                except NotProbabilistic as np:
                    continue

        for featureIndex in range(len(featureSubsets)):
            metrics.average(featureSubsets[featureIndex],
                            i,
                            cotrainer.classifiers[featureIndex])

        if 'combined' in metrics.graphNames:
            metrics.average('combined', i, cotrainer)

        i+=1
        
#print(f_measures)
#print(accuracies)
except KeyboardInterrupt:
    print('Terminating on keyboard interrupt')
except Exception as e:
    print('Terminating on unknown exception {}'.format(e))

#plt.plot(range(len(f_measures[0][0])), f_measures[0][0], 'r',
#         range(len(f_measures[0][1])), f_measures[0][1], 'g',
#         range(len(f_measures[1][0])), f_measures[1][0], 'b',
#         range(len(f_measures[1][1])), f_measures[1][1], 'y')

print(metrics.f_measures)
metrics.plotFmeasure()

learningType = 'cotrain'
if args.selftrain:
    learningType = 'selftrain'
metrics.savePlot('{}_{}_{}_{}_{}_{}.png'.format(learningType,
                                                args.classifier,
                                                args.positive,
                                                args.negative,
                                                args.unlabeled,
                                                args.numFolds))
