import nltk
from collections import defaultdict
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet  as wn
from nltk.corpus import stopwords

def wordnet_distance(word1, word2):
    maxy = 0
    for pos in wn.VERB,wn.NOUN,wn.ADJ,wn.ADV:
        try:
            word1synset = wn.synset('{}.{}.01'.format(word1, pos))
        except WordNetError:
            continue

        try:
            word2synset = wn.synset('{}.{}.01'.format(word2, pos))
        except WordNetError:
            continue

        r = word1synset.path_similarity(word2synset)
        if r is not None and r > maxy:
            maxy = r

    return maxy

from extractSentences import CausalityScorer
class FeatureExtractor:
    def __init__(self):
        with open('/home/chidey/PDTB/Discourse/config/markers/markers') as f:
            self.markers = f.read().splitlines()
        #self.cs = CausalityScorer()
        self.validFeatures = {'curr_stem' : self.getCurrStemNgrams,
                              'prev_stem' : self.getPrevStemNgrams,
                              #'reporting' : self.getReporting,
                              #'final_reporting' : self.getFinalReporting,
                              'coref' : self.getCoref,
                              #'head verb' : self.getHeadVerb,
                              #'pronoun' : self.getPronoun,
                              'intersection' : self.getIntersection,
                              'altlex_pos' : self.getAltlexPosNgrams,
                              'altlex_marker' : self.getAltlexMarker,
                              }

        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.iteritems())

    def getNgrams(self, featureName, gramList, n=(1,2)):
        features = {}
        prev = None
        for curr in gramList:
            if 1 in n:
                features['uni: ' + featureName + ': ' + curr] = True
            if prev:
                if 2 in n:
                    features['bi: ' + featureName + ': ' + prev + ' ' + curr] = True
            prev = curr

        return features

    def getCurrStemNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getCurrStemNgrams],
                              dataPoint.getCurrStem())

    def getPrevStemNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getPrevStemNgrams],
                              dataPoint.getPrevStem())

    def getCoref(self, dataPoint):
        altlexLower = dataPoint.getAltlexLower()
        if 'this' in altlexLower or 'that' in altlexLower:
            coref = True
        else:
            coref = False
            
        return {self.functionFeatures[self.getCoref]:
                coref}

    def getIntersection(self, dataPoint):
        inter = len(set(dataPoint.getCurrStem()) & set(dataPoint.getPrevStem()))
        return {self.functionFeatures[self.getIntersection]:
                inter}
            
    def getAltlexPosNgrams(self, dataPoint):
        return self.getNgrams(self.functionFeatures[self.getAltlexPosNgrams],
                              dataPoint.getAltlexPos())

    def getAltlexMarker(self, dataPoint):
        features = {}
        altlexLower = dataPoint.getAltlexLower()
        altlexLowerString = ' '.join(altlexLower)
        for marker in self.markers:
            if marker in altlexLower or len(marker.split()) > 1 and marker in altlexLowerString:
                value = True
            else:
                value = False
                
            features[self.functionFeatures[self.getAltlexMarker] + ' ' + marker] = value

        return features
    
    def addFeatures(self, dataPoint, featureSettings):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        features = {}
        for featureName in featureSettings:
            if featureName in self.validFeatures:
                
                features.update(self.validFeatures[featureName](dataPoint))
        return features


class Features:
    stop = stopwords.words("english")
    
    def __init__(self, corpus, n=2000):
        self.all_words = nltk.FreqDist(w.lower() for w in corpus.words)
        self.word_features = [i[0] for i in self.all_words.most_common(n)] #self.all_words.keys()[:2000]
        self.causatives = {'because', 'cause', 'after', 'as', 'since'}
        self.corpus = corpus
        x = defaultdict(int)
        for di in self.corpus.di:
            if di.relation == 'AltLex' and '.Cause.' in di.klass:
                for w in set(di.connective):
                    x[w] += 1
        self.altlexwords = set(i for i in x if x[i] >= 2)

    def unigram_features(self, document):
        document_words = set(document.all_words)
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def causal_features(self, document):
        document_words = set(document.all_words)
        return {'causal' : any(document_words & self.causatives)}

    def wordnet_causal_features(self, document, altlexes):
        document_words = set(document.secondarg)
        features = {}
        for word in self.altlexwords:
            #features['altlex (%s)' % word] = max(wordnet_distance(word, i) for i in document_words) #
            features['altlex (%s)' % word] = (word in document_words)
        for word in altlexes: #('mean', 'reason', 'result', 'cause'):
            features ['wordnet %s' % word] = max(wordnet_distance('cause', i) for i in document_words) 
            
        return features #ADJ ADV NOUN VERB
    
    def all_features(self, document, altlexes):
        features = self.unigram_features(document)
        features.update(self.wordnet_causal_features(document, altlexes))
        features.update(self.causal_features(document))
        return features

if __name__ == '__main__':
    import pdtbsql as p

    cb = p.CorpusBuilder()
    c = cb.extract()
    uf = Features(c)

    #balanced training
    pos = [i for i in c.di if '.Cause.' in i.klass and i.relation not in ('Explicit','AltLex')]
    neg = [i for i in c.di if '.Cause.' not in i.klass and i.relation not in ('Explicit','AltLex')][:len(pos)]

    for altlex in ({'mean','reason','result','cause'},{}):#({'mean'},{'reason'},{'result'},{'cause'},{}):
        print(altlex)
        featuresets = [(uf.all_features(d,altlex), 'Cause' in d.klass) for d in pos+neg]
        train_set = featuresets[:int(len(pos)/2)]+featuresets[-int(len(pos)/2):]
        test_set = featuresets[int(len(pos)/2):-int(len(pos)/2)]

        classifier = nltk.NaiveBayesClassifier.train(train_set)
    #find most indicative features
    #classifier.show_most_informative_features(25)

        print(nltk.classify.accuracy(classifier, test_set))
    #0.5776237304530066
    #0.59 with proper splitting and tokenizing
    #0.603256488795744 with tokenizing and causatives
    #0.5680206318504191 with snowball stemmer and stop words
    #0.5978401031592521 with just tokenization
    #0.6017085751128304 adding altlex
    #0.6047711154094133 adding causal indicator
    #0.6073500967117988 with WN path sim to 'cause'
    #0.6052546744036106 adding lemmatization
    #0.6120245003223727 with WN path similarity to cause, reason, result, mean
    #0.6091231463571889 worse without lemmatization (needed for path similarity)
    #0.6052546744036106 adding explain,divert,vary,assess,perceive

    #ignoring explicit,altlex
    #with similarity
    #0.6028696498054474
    #without similarity ... disappointing
    #0.605544747081712
