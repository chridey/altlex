import os
import json
import collections
import math

from sklearn.externals import joblib

from altlex.featureExtraction.featureExtractor import FeatureExtractor,filterFeatures,makeInteractionFeatures
from altlex.featureExtraction import config
from altlex.featureExtraction.dataPoint import makeDataPoint, makeDataPointsFromAltlexes

class AltlexHandler:
    def __init__(self,
                 featureSettings=None,
                 classifierFile=None,
                 altlexFile=None,
                 verbose=False):

        if featureSettings is None:
            featureSettings = config.defaultConfig

        self.featureExtractor = FeatureExtractor(featureSettings,
                                                 verbose)

        self._classifierFile = classifierFile
        self._classifier = None #best classifier trained on wikipedia, load from settings
        self._vectorizer = None
        
        self.altlexFile = altlexFile
        self._causalAltlexes = None #causal altlexes, load from settings
        self._nonCausalAltlexes = None
        
    def loadAltlexes(self):
        if self.altlexFile is None:
            self.altlexFile = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),
                                                            '..'), 'config'), 'altlexes.json')
            
        with open(self.altlexFile) as f:
            altlexes = json.load(f)

        self._causalAltlexes = altlexes['causal']
        self._nonCausalAltlexes = altlexes['noncausal']

    @property
    def classifierFile(self):
        if self._classifierFile is None:
            self._classifierFile = os.path.join(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),
                                                            '..'), 'config'), 'models'),
                                                            'full_plus_sgd_st1_inter1_unbalanced_combined_bootstrap.1')
                                                            
        return self._classifierFile
    
    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = joblib.load(self.classifierFile)
        return self._classifier
    
    @property
    def vectorizer(self):
        if self._vectorizer is None:
            self._vectorizer = joblib.load(self.classifierFile + '.vectorizer')
        return self._vectorizer

    @property
    def causalAltlexes(self):
        if self._causalAltlexes is None:
            self.loadAltlexes()
        return self._causalAltlexes          

    @property
    def nonCausalAltlexes(self):
        if self._nonCausalAltlexes is None:
            self.loadAltlexes()
        return self._nonCausalAltlexes          

    def dataPoints(self, sentence):
        dataPoints = []
        
        for dataPoint in makeDataPointsFromAltlexes(sentence, self.causalAltlexes, True):            
            if dataPoint.altlexLength:
                dataPoints.append(dataPoint)

        return dataPoints
            
    def addFeatures(self, metadataList, interaction=True):
        featuresList = []
        for dataPoint in metadataList:
            features = self.featureExtractor.addFeatures(dataPoint)
            print(dataPoint.getAltlex())
            #add interaction features
            if interaction:
                filtered_features = filterFeatures(features,
                                                   None,
                                                   ['kld', 'framenet', 'altlex'])
                interaction_features = makeInteractionFeatures(filtered_features,
                                                               'prev',
                                                               'curr')
                features.update(interaction_features)

            featuresList.append(features)
            
        return featuresList
    
    def predict(self, metadataList, interaction=True):
        if not len(metadataList):
            return []
        
        featuresList = self.addFeatures(metadataList, interaction)
        
        try:
            features_transformed = self.vectorizer.transform(featuresList)
        except ValueError as e:
            print("problem with feature transformer: {}".format(e))
            return [0]*len(metadataList)

        return self.classifier.predict(features_transformed)
    
