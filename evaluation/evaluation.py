import argparse
import numpy as np

from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.externals import joblib

from altlex.featureExtraction.dataPointMetadata import DataPointMetadataList

class MostCommonClassEvaluator:
    def __init__(self, train, combined=False):
        self.train = DataPointMetadataList.load(train)
        if combined:
            self.altlexes = self.train.combinedAltlexes
        else:
            self.altlexes = self.train.altlexes

        self.totals = [1.*sum(self.altlexes[j].values()) for j in self.altlexes]
        
    def predict(self, test, dedupe=False):
        if dedupe:
            test = test.dedupe(self.train, True)

        y_pred = []
        for index,datum in enumerate(test):
            score,prediction = max(((self.altlexes[i][tuple(datum.altlex)]/self.totals[i],
                                     i) for i in range(len(self.altlexes))),
                                   key=lambda x:x[0])
            y_pred.append(prediction)

        return y_pred
    
def getResults(labels, predictions, combined=False):
    a = accuracy_score(labels, predictions)
    if combined:
        smallestClass = 1##0 if sum(labels) > len(labels) - sum(labels) else 1
        p,r,f,s = precision_recall_fscore_support(labels, predictions,
                                                  average='binary', pos_label = smallestClass)
    else:
        p,r,f,s = precision_recall_fscore_support(labels, predictions)

    return a,p,r,f

def printResults(a, p, r, f, format=None, prefix=''):
    if format is None:
        print ("accuracy: {} precision: {} recall: {} f-score: {}".format(a, p, r, f))
    elif format == 'latex':
        print ("{} & {} & {} & {} & {} \\".format(prefix, a, p, r, f))
        
def mostCommonClassEvaluator(altlexes, test, combined=False, format=None, prefix=''):
    totals = [1.*sum(altlexes[j].values()) for j in altlexes]
    print(totals)

    y_pred = []
    test_labels = np.array(list(test.iterLabels(combined=combined)))

    for index,datum in enumerate(test):
        score,prediction = max(((altlexes[i][tuple(datum.altlex)]/totals[i],
                                 i) for i in range(len(altlexes))),
                               key=lambda x:x[0])
        y_pred.append(prediction)

    a,p,r,f = getResults(test_labels, y_pred, combined=combined)
    printResults(a, p, r, f, format, prefix)
                                                        
def makeNgramOnlyFeatures(j):
    new_j = []

    for f,label in j:
        features = {}
        altlex_ngram = f['altlex'][:len(f['altlex'])/2]
        for i in range(len(altlex_ngram)-1):
            features['altlex_stem_' + altlex_ngram[i]] = 1
            features['altlex_stem_' + altlex_ngram[i] + '_' + altlex_ngram[i+1]] = 1
        features['altlex_stem_' + altlex_ngram[-1]] = 1
        new_j.append((features,label))

    return new_j

def evaluate(test, classifier, feature_vectorizer, combined=False, format=None, prefix=''):
    test_features = feature_vectorizer.transform(data.features for data in test)
    test_labels = np.array(list(test.iterLabels(combined=combined)))

    y_pred = classifier.predict(test_features)
    
    a,p,r,f = getResults(test_labels, y_pred, combined=combined)

    printResults(a, p, r, f, format, prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate a classifier on a dataset with altlexes')

    parser.add_argument('testfile', 
                        help='the training metadata in gzipped JSON format')
    parser.add_argument('classifier', 
                        help='the joblib.dump ed classifier')
    parser.add_argument('--combined',
                        action='store_true')
    parser.add_argument('--common',
                        action='store_true')
    parser.add_argument('--prefix')
    parser.add_argument('--format')

    args = parser.parse_args()

    test = DataPointMetadataList.load(args.testfile)
    
    if args.common:
        train = DataPointMetadataList.load(args.classifier)
        if args.combined:
            altlexes = train.combinedAltlexes
        else:
            altlexes = train.altlexes
        test = test.dedupe(train, True)
        mostCommonClassEvaluator(altlexes, test, combined=args.combined,
                                 format=args.format, prefix=args.prefix)
    else:
        classifier = joblib.load(args.classifier)
        vectorizer = joblib.load(args.classifier + '.vectorizer')

        evaluate(test, classifier, vectorizer, combined=args.combined,
                                 format=args.format, prefix=args.prefix)

