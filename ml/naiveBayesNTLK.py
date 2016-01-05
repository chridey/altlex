import math

import nltk

from chnlp.ml.sklearner import Sklearner

class NaiveBayes(Sklearner):
    def _transformOne(self, featureSet):
        transformedFeatureSet = {}

        for featureName in featureSet:
            if type(featureSet[featureName]) in (int, float):
                if featureSet[featureName] > 0:
                    transformedFeatureSet[featureName] = \
                                        math.ceil(math.log(featureSet[featureName],2))
                else:
                    transformedFeatureSet[featureName] = -100
            else:
                transformedFeatureSet[featureName] = featureSet[featureName]

        return transformedFeatureSet

    def _transform(self, featureSets):
        #heuristically transform numerical features into categorical features
        #if less than 1 multiply by 100 and take the log2-ceil
        #if greater than 1 just take the log2-ceil
        transformedFeatureSets = []
        for featureSet,label in featureSets:
            transformedFeatureSets.append((self._transformOne(featureSet),label))

        return Sklearner.transform(self, transformedFeatureSets)
        
    def train(self, training):
        self.model = nltk.NaiveBayesClassifier.train(self._transform(training))
        return self.model
    
    def classify(self, featureSet):
        return self.model.classify(self._transformOne(featureSet))

    def accuracy(self, testing):
        return nltk.classify.accuracy(self.model, self._transform(testing))

    def show_most_informative_features(self, n=50):
        return self.model.show_most_informative_features(n)

    def save(self, filename):
        pass
