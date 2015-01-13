#read in JSON data set
#set aside 30% of data (~200 causal examples) for testing
#oversample remaining causal examples
#add features

import json
import argparse
import itertools

from chnlp.utils.utils import splitData
from chnlp.altlex.featureExtractor import makeDataset
from chnlp.altlex.config import Config

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
parser.add_argument('--numFolds', type=int, default=2,
                    help='the number of folds for crossvalidation (default: %(default)s)')

parser.add_argument('--test', '-t', action='store_true',
                    help = 'test on the set aside data (default: train only)')

parser.add_argument('--save', metavar = 'S',
                    help = 'save the model to a file named S')

parser.add_argument('--config',
                    help = 'JSON config file to use instead of command line options')

args = parser.parse_args()

#start with the config file options, but allow them to be overwritten by command line
if args.config:
    config.setParams(json.load(args.config))

classifierType = config.classifiers[args.classifier]

with open(args.infile) as f:
    data = json.load(f)

#first create dataset and assign features
settingKeys = list(config.experimentalSettings)

for settingValues in itertools.product((True,False),
                                       repeat=len(settingKeys)):
    featureSettings = config.fixedSettings.copy()
    featureSettings.update(dict(zip(settingKeys, settingValues)))
        
    print(featureSettings)

    causalSet, nonCausalSet = makeDataset(data,
                                          config.featureExtractor,
                                          featureSettings)

    print("True points: {} False points: {}".format(len(causalSet),
                                                    len(nonCausalSet)))

    training,testing = splitData(causalSet, nonCausalSet)
    #print(len(training), numNonCausalTraining/len(training),
    #      len(testing), numNonCausalTesting/len(testing))

    classifier = classifierType()

    if args.crossvalidate:
        classifier.crossvalidate(training, n_folds=args.numFolds)
    else:
        classifier.train(training)

    classifier.show_most_informative_features(50)

    if args.test:
        accuracy = classifier.accuracy(testing)
        precision, recall = classifier.metrics(testing)
        classifier.printResults(accuracy, precision, recall)

if args.save:
    classifier.save(args.save)
