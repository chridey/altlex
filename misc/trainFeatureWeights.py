import argparse
import time
import json

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier

from altlex.utils import utils,wordUtils
from altlex.featureExtraction import featureExtractor
from altlex.featureExtraction.dataPointMetadata import DataPointMetadataList
from altlex.evaluation.annotations.causalAnnotations import getCausalAnnotations

from sklearn.preprocessing import MaxAbsScaler

from altlex.ml.gridSearch import HeldOutGridSearch
from altlex.ml.miniBatchSGD import MiniBatchSGD
from altlex.ml import sklearner

from altlex.evaluation.evaluation import evaluate

from sklearn.naive_bayes import BernoulliNB


def main(train, classifier, test=None, tune=None, balance=False, combined=False):
    #balance the data
    if balance:
        train_balanced,train_labels = zip(*utils.balance(zip(train.iterFeatures(),
                                                             train.iterLabels(combined=combined)),
                                                         oversample=15000,
                                                         verbose=True))
    else:
        train_balanced = train.iterFeatures()
        train_labels = train.iterLabels(combined=combined)

    feature_vectorizer = DictVectorizer()
    train_features = feature_vectorizer.fit_transform(train_balanced)
    train_labels = np.array(list(train_labels))

    print(train_features.shape)
    print(train_labels.shape)

    if tune:
        tune_features = feature_vectorizer.transform(data.features for data in tune)
        tune_labels = np.array(list(tune.iterLabels(combined=combined)))
        classifier.setTuning((tune_features, tune_labels))

    try:
        classifier.fit(train_features, train_labels)
    except KeyboardInterrupt:
        print('Terminating on keyboard interrupt')

    if test:
        evaluate(test, classifier, feature_vectorizer, combined)
        
    return classifier,feature_vectorizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train and/or evaluate a classifier on a dataset with altlexes')

    parser.add_argument('trainfile', 
                        help='the training metadata in gzipped JSON format')
    parser.add_argument('--tune',
                        type=float,
                        help='percentage of training to set aside for tuning')
    parser.add_argument('--testdir',
                        help='annotated test directory to extract altlexes from')
    parser.add_argument('--testfile',
                        help='test data in gzipped JSON format')
    parser.add_argument('--tunefile',
                        help='tune data in gzipped JSON format')
    parser.add_argument('--config',
                        help='config file for feature extractor in JSON format')
    parser.add_argument('--start',
                        type = int,
                        choices = (0,1),
                        default = 1)
    parser.add_argument('--batch_size',
                        type = int,
                        default = 100)
    parser.add_argument('--num_epochs',
                        type = int,
                        default = 50)
    parser.add_argument('--search_parameters',
                        type=json.loads,
                        default={'penalty': ['l2'],
                                 'alpha': [0.0001]})
    parser.add_argument('--parameters',
                        type=json.loads,
                        default={'penalty': 'l2',
                                 'alpha': 0.0001})
    
    parser.add_argument('--filter')
    parser.add_argument('--ablate')    
    parser.add_argument('--interaction',
                        action='store_true')

    parser.add_argument('--balance',
                        action='store_true')
    parser.add_argument('--combined',
                        action='store_true')

    parser.add_argument('--save')
    parser.add_argument('--load')    

    parser.add_argument('-v', '--verbose',
                        action='store_true')

    args = parser.parse_args()
    
    dataset = DataPointMetadataList.load(args.trainfile)
    print(len(dataset))
    
    labelLookup = wordUtils.trinaryCausalSettings[1]
    
    if args.tune:
        indicesList = dataset.split(args.tune)
    else:
        indicesList = [dataset.datumIndices]

    datasets = dataset.subsets(*indicesList)

    if args.tunefile:
        orig = DataPointMetadataList.load(args.tunefile)
        datasets.append(orig.dedupe(dataset, True))

    #extract test set here
    if args.testdir:
        if args.config:
            with open(args.config) as f:
                settings = json.load(f)
            fe = featureExtractor.FeatureExtractor(settings, verbose=True)
        else:
            fe = featureExtractor.FeatureExtractor(verbose=True)

        orig = getCausalAnnotations(args.testdir,
                                    datasets[0].causalAltlexes,
                                    fe,
                                    labelLookup)
        datasets.append(orig.dedupe(dataset, True))
    elif args.testfile:
        orig = DataPointMetadataList.load(args.testfile)
        datasets.append(orig.dedupe(dataset, True))

    if args.start == 0:
        #set to just explicit connectives
        #set features to just context features
        #save other indices for later
        datasets = [dataset.withConnectiveOnly(labelLookup) for dataset in datasets]
            
    for dataset in datasets:
        for data in dataset:
            '''
            if args.interaction:
                interactionSettings = {'include': None,
                                       'ablate': ['kld', 'framenet', 'altlex'],
                                       'first': 'prev',
                                       'second': 'curr'}
            else:
                interactionSettings = None

            if args.start == 0:
                ablate = args.ablate + ',' if args.ablate else ''
                data.features = createFeatureSet(data.features,
                                                args.filter,
                                                ablate + 'kld,altlex',
                                                interactionSettings)
                                             
            data.features = createFeatureSet(data.features,
                                             args.filter,
                                             args.ablate,
                                             interactionSettings)
            '''
            if args.filter:
                features = featureExtractor.filterFeatures(data.features,
                                                           args.filter.split(','),
                                                           None)
            else:
                features = data.features

            if args.ablate:
                features = featureExtractor.filterFeatures(features,
                                                           None,
                                                           args.ablate.split(','))
                
            if args.interaction:
                filtered_features = featureExtractor.filterFeatures(features,
                                                                    None,
                                                                    ['kld', 'framenet', 'altlex'])
                interaction_features = featureExtractor.makeInteractionFeatures(filtered_features,
                                                                                'prev',
                                                                                'curr')
                if args.start == 0:
                    data.features = filtered_features

                features.update(interaction_features)

            data.features = features

    #balance the data
    if args.balance:
        train_balanced,labels = zip(*utils.balance(zip(datasets[0].iterFeatures(),
                                                       datasets[0].iterLabels(combined=args.combined)),
                                                   oversample=15000,
                                                   verbose=True))
    else:
        train_balanced = datasets[0].iterFeatures()
        labels = datasets[0].iterLabels(combined=args.combined)

    feature_vectorizer = DictVectorizer()
    features = [feature_vectorizer.fit_transform(train_balanced)]
    features.extend([feature_vectorizer.transform(data.features for data in dataset) for dataset in datasets[1:]])

    labels = [np.array(list(labels))] + [np.array(list(dataset.iterLabels(combined=args.combined))) for dataset in datasets[1:]]

    print([feature.shape for feature in features])
    print([label.shape for label in labels])

    test = [(features[i], labels[i]) for i in range(1, len(features))]
    parameters = dict(n_iter=args.num_epochs,
                      shuffle=True,
                      batch_size=args.batch_size,
                      verbose=args.verbose,
                      test=test)

    if args.tune or args.tunefile:
        parameters['verbose'] = False
        classifier = HeldOutGridSearch('mini_batch_sgd',
                                       parameters,
                                       args.search_parameters,
                                       verbose=args.verbose,
                                       transformer=sklearner.Identity(),
                                       tuning=(features[1], labels[1])
                                       )
    else:
        #parameters['penalty'] = 'elasticnet' #'l2' #
        #parameters['alpha'] =  0.0001 #.00001 #
        parameters.update(args.parameters)
        classifier = MiniBatchSGD(**parameters)
        #classifier = SGDClassifier(n_iter=args.num_epochs)
        #classifier = BernoulliNB()

    if args.load:
        classifier = joblib.load(args.load)

    try:
        classifier = classifier.fit(features[0], labels[0])
    except KeyboardInterrupt:
        print('Terminating on keyboard interrupt')
        
    for index in range(1, len(features)):
        y_pred = classifier.predict(features[index])
        p,r,f,s = precision_recall_fscore_support(labels[index], y_pred)
        print ("precision: {} recall: {} ".format(p, r))

    if args.save:
        joblib.dump(classifier, args.save)
        joblib.dump(feature_vectorizer, args.save + '.vectorizer')
        datasets[-1].save(args.save + '.data')
