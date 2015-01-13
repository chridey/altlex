import json
import argparse
import collections

#import matplotlib
import matplotlib.pyplot as plt

from chnlp.ml.cotrainer import Cotrainer, NotProbabilistic
from chnlp.utils.utils import splitData, sampleDataWithoutReplacement
from chnlp.altlex.featureExtractor import makeDataset
from chnlp.altlex.config import Config

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

parser.add_argument('--unlabeled', '-u', metavar='U', type=int, default=75,
                    help='the number of unlabeled examples to sample at random (default: %(default)s) (P + N must be less than U)')

parser.add_argument('--iterations', '-k', metavar='K', type=int, default=30,
                    help='the number of iterations (default: %(default)s)')

parser.add_argument('--numFolds', type=int, default=2,
                    help='the number of folds for crossvalidation (default: %(default)s)')

parser.add_argument('--classifier', '-c', metavar='C',
                    choices = config.classifiers.keys(),
                    default = config.classifier,
                    help = 'the supervised learner to use (default: %(default)s) (choices: %(choices)s)')

parser.add_argument('--save', metavar = 'S',
                    help = 'save the model to a file named S')

args = parser.parse_args()

classifiers = [config.classifiers[args.classifier](),
               config.classifiers[args.classifier]()]
cotrainer = Cotrainer(*classifiers)

#classifiers = [config.classifiers['random_forest'](),
#               config.classifiers['svm']()]

with open(args.taggedFile) as f:
    taggedData = json.load(f)

with open(args.untaggedFile) as f:
    untaggedData = json.load(f)

dataLookup = collections.defaultdict(dict)
featureSubsets = (('semantic',),
                  ('syntactic', 'lexical', 'structural'))
#featureSubsets = (('semantic', 'syntactic', 'lexical', 'structural'),
#                  ('semantic', 'syntactic', 'lexical', 'structural'))

#first create the metadata and split the datasets for the tagged data
for index,featureSubset in enumerate(featureSubsets):
    causalSet, nonCausalSet = makeDataset(taggedData,
                                          config.featureExtractor,
                                          config.featureSubset(*featureSubset))
    evaluation, testing = splitData(causalSet, nonCausalSet)

    #add these to some lookup table, blah blah
    dataLookup['tagged'][featureSubset] = []
    for training, testing in classifiers[index].iterFolds(evaluation,
                                                          n_folds=args.numFolds):

        dataLookup['tagged'][featureSubset].append((training, testing))

#now randomly get U examples from the untagged data
for featureSubset in featureSubsets:
    emptySet, unknownSet = makeDataset(untaggedData,
                                       config.featureExtractor,
                                       config.featureSubset(*featureSubset),
                                       #max=args.unlabeled
                                       )
    #sample here
    startingData, remainingData = sampleDataWithoutReplacement(unknownSet, args.unlabeled)
    dataLookup['untagged'][featureSubset] = startingData #,remainingData

#f_measures = {i: collections.defaultdict(list) for i in range(args.numFolds)}
#accuracies = {i: collections.defaultdict(list) for i in range(args.numFolds)}

f_measures = collections.defaultdict(list)

try:
    for i in range(args.iterations):
        print('iteration {}'.format(i))

        accuracies = collections.defaultdict(list)
        precisions = collections.defaultdict(list)
        recalls = collections.defaultdict(list)
        
        for j in range(args.numFolds):
            print('fold {}'.format(j))

            combinedTesting = []
            for k,featureSubset in enumerate(featureSubsets):
                print('features {}'.format(featureSubset))

                training = dataLookup['tagged'][featureSubset][j][0]
                print('training is {}'.format(len(training)))
                classifiers[k].train(training)

                numPositives = 0
                numNegatives = 0
                index = 0
                while numPositives < args.positive or numNegatives < args.negative:
                    dataPoint = dataLookup['untagged'][featureSubset][index][0]
                    label = classifiers[k].classify(dataPoint)

                    append = False
                    if label:
                        if numPositives < args.positive:
                            numPositives += 1
                            append = True
                    else:
                        if numNegatives < args.negative:
                            numNegatives += 1
                            append = True

                    if append:
                        #print(dataPoint)
                        #print(label)
                        dataLookup['tagged'][featureSubset][j][0].append((dataPoint,
                                                                       label))
                    index +=1

                dataLookup['untagged'][featureSubset] = \
                                            dataLookup['untagged'][featureSubset][index:]

                testing = dataLookup['tagged'][featureSubset][j][1]
                accuracy = classifiers[k].accuracy(testing)
                precision, recall = classifiers[k].metrics(testing)

                accuracies[k].append(accuracy)
                precisions[k].append(precision)
                recalls[k].append(recall)
                #f_measure = classifiers[k].printResults(accuracy, precision, recall)
                #f_measures[j][k].append(f_measure)
                #accuracies[j][k].append(accuracy)
                combinedTesting.append(testing)

            print('calculating combined metrics')
            accuracy = 0 #TODO
            try:
                precision,recall = cotrainer.metrics(combinedTesting)
            except NotProbabilistic as np:
                continue
            accuracies['overall'].append(accuracy)
            precisions['overall'].append(precision)
            recalls['overall'].append(recall)

        f_measures['overall'].append(cotrainer.printResults(accuracies['overall'],
                                                            precisions['overall'],
                                                            recalls['overall']))

                                                            
        for k in range(len(featureSubsets)):
            f_measures[k].append(classifiers[k].printResults(accuracies[k],
                                                             precisions[k],
                                                             recalls[k]))

        
    #train a classifier on semantic features
    #train a classifier on all the other features

    #use each of those classifiers to label the random data from the untagged data

    #add these examples to the training set
    #replenish the untagged data
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
plt.plot(range(len(f_measures[0])), f_measures[0], 'r',
         range(len(f_measures[1])), f_measures[1], 'b',
         range(len(f_measures['overall'])), f_measures['overall'], 'y',
         )

plt.savefig('cotrain_{}_{}_{}_{}_{}.png'.format(args.classifier,
                                                args.positive,
                                                args.negative,
                                                args.unlabeled,
                                                args.numFolds))
