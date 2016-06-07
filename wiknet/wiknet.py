import collections
import operator

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

from altlex.wiknet import wikipedia
from altlex.ml.sklearner import Sklearner

def isContentWord(pos):
    return pos[:2] in ('NN', 'VB', 'JJ', 'RB', 'CD')

def iterDocs(parsedData):
    for article in parsedData:
        lemmas = []
        for wikipedia in article['sentences']:
            for sentence in wikipedia:
                lemmas += reduce(operator.add, sentence['lemmas'])
        yield lemmas
                
def calcTFIDF(parsedData):
    tfidf = TfidfVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)

    tfidf.fit(iterDocs(parsedData))
    
    return {i:tfidf.idf_[tfidf.vocabulary_[i]] for i in tfidf.vocabulary_}

def getLemmas(lemmas):
    return reduce(operator.add, lemmas)

class WikNetMatch:
    def __init__(self, filename='/proj/nlp/users/chidey/simplification/WikNet_word_pairs',
                 scores = None,
                 idf = None):
        if scores is not None:
            self.scores = scores
        else:
            self.scores = collections.defaultdict(dict)
            with open(filename) as f:
                for line in f:
                    word1, word2, score = line.strip().split()
                    self.scores[word1][word2] = float(score)
        self.idf = idf
        
    def __call__(self, a, b, penalty=0, idf=None, verbose=False):
        return self.matchSentence(a, b, penalty)
    
    def matchSentence(self, a, b, penalty=0, structural=True, verbose=False, reverse=False):
        if reverse:
            tmp = a
            a = b
            b = tmp
            
        score = 0
        aMatches = set()
        bMatches = set()

        aLemmas = reduce(operator.add, a['lemmas'])
        bLemmas = reduce(operator.add, b['lemmas'])
        aPos = reduce(operator.add, a['pos'])
        bPos = reduce(operator.add, b['pos'])

        #calculate mapping from dependent to governor
        #TODO: move this outside so that we dont calc it every time
        if structural:
            aDepToGov = collections.defaultdict(dict)
            bDepToGov = collections.defaultdict(dict)
            for i,deps in enumerate(a['dep']):
                for triple in deps:
                    gov, dep, rel = triple
                    aDepToGov[dep + (len(a['lemmas'][i-1]) if i > 0 else 0)] = (gov, rel)
            for i,deps in enumerate(b['dep']):
                for triple in deps:
                    gov, dep, rel = triple
                    bDepToGov[dep + (len(b['lemmas'][i-1]) if i > 0 else 0)] = (gov, rel)

        numContent = 0
        totalIDF = 0
        for i1,word1 in enumerate(aLemmas):
            maxSigma = 0
            argmax = None
            if not isContentWord(aPos[i1]):
                continue
            numContent += 1
            for i2,word2 in enumerate(bLemmas):
                if not isContentWord(bPos[i2]):
                    continue
                sigma = self.matchWord(word1, word2)
                #add in dependency score
                if structural:
                    if i1 in aDepToGov and i2 in bDepToGov:
                        if aDepToGov[i1][1] == bDepToGov[i2][1]:
                            #if the words have the same dependency relation
                            if aDepToGov[i1][1] == 'root':
                                sigma += .5
                            else:
                                sigma += .5*self.matchWord(aLemmas[aDepToGov[i1][0]],
                                                           bLemmas[bDepToGov[i2][0]])
                    elif word1 == word2: sigma += .5
                
                if sigma > maxSigma:
                    maxSigma = sigma
                    argmax = i2

            if verbose:
                if argmax is not None:
                    print(word1, maxSigma, bLemmas[argmax])
                else:
                    print(word1, maxSigma, None)
            
            if self.idf:
                maxSigma *= self.idf[word1]
                totalIDF += self.idf[word1]
            if maxSigma:
                aMatches.add(i1)
                bMatches.add(argmax)
            score += maxSigma

        #calculate normalizer (maximum possible score for a)
        if self.idf:
            Z =  (1+structural*.5)*totalIDF
        else:
            Z = (1+structural*.5)*numContent #len(aLemmas)
        if Z == 0:
            if verbose:
                print(' '.join(aLemmas), ' '.join(bLemmas))
            if penalty == 'both':
                return 0,0
            return 0

        if penalty:
            #find the content words that have no matches
            count = 0
            for i1 in set(range(len(aLemmas))) - aMatches:
                if isContentWord(aPos[i1]):
                    count += 1
            for i2 in set(range(len(bLemmas))) - bMatches:
                if isContentWord(bPos[i2]):
                    count += 1

            if penalty == 'only':
                return 1.*count/(count + len(aMatches) + len(bMatches))
            if penalty == 'both':
                return 1.*score/Z, 1.*count/Z
            score -= count * penalty

        return 1.*score/Z
            
    def matchWord(self, a, b):
        if a == b:
            return 1
        if a in self.scores and b in self.scores[a]:
            return self.scores[a][b]
        if b in self.scores and a in self.scores[b]:
            return self.scores[b][a]
        
        return 0

    def predict(self, parsedData, titles, penalty=None, structural=True, multisentence=False, reverse=False):
        parsedTitles = dict(zip([i['title'] for i in parsedData], range(len(parsedData))))
        assert(set(titles.keys()).issubset(set(parsedTitles.keys())))

        scores = []
        for title in sorted(titles.keys()):
            print(title)
            article = parsedData[parsedTitles[title]]['sentences']

            for a in article[0]:
                if a is None or not len(a['lemmas']):
                    scores.extend([0] * (len(article[1])))
                    if multisentence:
                        scores.extend([0] * (len(article[1])-1))
                    continue
                for b in article[1]:
                    if b is None or not len(b['lemmas']):
                        scores.append(0)
                    else:
                        scores.append(self.matchSentence(a, b, penalty=penalty, structural=structural, reverse=reverse))
                if multisentence:
                    for i in range(len(article[1])-1):
                        if article[1][i] is None or not len(article[1][i]['lemmas']) or not len(article[1][i+1]['lemmas']):
                            scores.append(0)
                        else:
                            b = {'lemmas': article[1][i]['lemmas'] + article[1][i+1]['lemmas'],
                                 'pos': article[1][i]['pos'] + article[1][i+1]['pos'],
                                 'dep': article[1][i]['dep'] + article[1][i+1]['dep']}
                            scores.append(self.matchSentence(a, b, penalty=penalty, structural=structural, reverse=reverse))
            if multisentence:
                for i in range(len(article[0])-1):
                    if article[0][i] is None or not len(article[0][i]['lemmas']) or not len(article[0][i+1]['lemmas']):
                        scores.extend([0] * (2*len(article[1])-1))
                        continue
                    a = {'lemmas': article[0][i]['lemmas'] + article[0][i+1]['lemmas'],
                         'pos': article[0][i]['pos'] + article[0][i+1]['pos'],
                         'dep': article[0][i]['dep'] + article[0][i+1]['dep']}
                    for b in article[1]:
                        if b is None or not len(b['lemmas']):
                            scores.append(0)
                        else:
                            scores.append(self.matchSentence(a, b, penalty=penalty, structural=structural, reverse=reverse))
                    scores.extend([0] * (len(article[1])-1))
            
        return scores

    def greedyPredict(self, parsedData, annotatedData, titles, penalty=None, scores=None, multisentence=False):
        if scores is None:
            scores = self.predict(parsedData, titles, penalty, multisentence)
        
        newScores = []
        index = 0
        for title in titles.keys():
            sents1 = annotatedData['articles'][0][titles[title]]
            sents2 = annotatedData['articles'][1][titles[title]]
            len1 = len(sents1)+multisentence*(len(sents1)-1)
            len2 = len(sents2)+multisentence*(len(sents2)-1)
            m = np.array(scores[index:index+len1*len2]).reshape((len1,
                                                                 len2))
            indices = [(i,j) for i in range(m.shape[0]) for j in range(m.shape[1])]
            pairs = wikipedia.getMaximalMatching(sents1,
                                                 sents2,
                                                 dict(zip(indices, m.ravel())))

            #print(len(dict(zip(indices, m.ravel()))))
            #print(len(pairs))
            labels = [(i,j) in pairs for i in range(m.shape[0]) for j in range(m.shape[1])]
            #print(sum(labels))
            newScores.extend((m.ravel()*labels).tolist())
                
            index += len1*len2

        return newScores
                                                 
def getTitles(annotatedData):
    return dict(zip(annotatedData['titles'], range(len(annotatedData['titles']))))

def getParsedTitles(parsedData):
    return {article['title']:i for i,article in enumerate(parsedData)}

def getLabels(annotatedData, titles):
    labels = []

    for title in sorted(titles):
        print(title)
        matches = annotatedData['matches'][titles[title]]
        m = np.array(matches)
        print(m.shape)
        labels.extend((m >= 3).ravel().tolist())
        '''
        for i1 in range(len(matches)):
            for i2 in range(len(matches[i1])):
                labels.append(int(matches[i1][i2]) >= 3)
        '''
    return labels

def evaluate(labels, scores, minThresh=.3, maxThresh=.9, step=.05, indices=Ellipsis):
    s = np.array(scores)
    l = np.array(labels)
    results = []
    for thresh in np.arange(minThresh, maxThresh, step):
        predictions = s[indices] > thresh
        precision, recall, f_score, support = precision_recall_fscore_support(l[indices],
                                                                              predictions,
                                                                              average='binary')
        results.append((float('%.2lf' % thresh), precision, recall, f_score))
    return results

def evaluateCombined(labels, scores1, scores2, minThresh=.3, maxThresh=.9, step=.05, indices=Ellipsis):
    s1 = np.array(scores1)
    s2 = np.array(scores2)    
    l = np.array(labels)
    results = []
    for thresh1 in np.arange(minThresh, maxThresh, step):
        predictions1 = s1[indices] > thresh1
        for thresh2 in np.arange(minThresh, maxThresh, step):
            predictions2 = s2[indices] > thresh2
            predictions = predictions1 + predictions2
            precision, recall, f_score, support = precision_recall_fscore_support(l[indices],
                                                                                  predictions,
                                                                                  average='binary')
            results.append((float('%.2lf' % thresh1),
                            float('%.2lf' % thresh2),
                            precision, recall, f_score))
    return results

def getLengthDiffs(parsedData, titles, multisentence=False):
    ret = []

    for title in sorted(titles.keys()):
        #print(title)
        article = parsedData[titles[title]]['sentences']

        for a in article[0]:
            for b in article[1]:
                ret.append(abs(len(getLemmas(a['lemmas'])) - len(getLemmas(b['lemmas']))))
                
            if multisentence:
                for i in range(len(article[1])-1):
                    b = article[1][i]['lemmas'] + article[1][i+1]['lemmas']
                    ret.append(abs(len(getLemmas(a['lemmas'])) - len(getLemmas(b))))
        if multisentence:
            for i in range(len(article[0])-1):
                a = article[0][i]['lemmas'] + article[0][i+1]['lemmas']
                for b in article[1]:
                    ret.append(abs(len(getLemmas(a)) - len(getLemmas(b['lemmas']))))

                ret.extend([0] * (len(article[1])-1))
            
    return ret
            
def getIndices(parsedData, titles, one2one=True, one2two=False, two2one=False, two2two=False, validTitles=None):

    ret = []
    index = 0
    for title in sorted(titles):
        article = parsedData[titles[title]]
        if validTitles is not None and title not in validTitles:
            index += len(article['sentences'][0])+len(article['sentences'][1])-2
            continue
        for i in range(len(article['sentences'][0])):
            if one2one:
                ret.extend(list(range(index, index+len(article['sentences'][1]))))
            index += len(article['sentences'][1])
            if one2two:
                ret.extend(list(range(index, index+len(article['sentences'][1])-1)))
            index += len(article['sentences'][1])-1

        for i in range(len(article['sentences'][0])-1):
            if two2one:
                ret.extend(list(range(index, index+len(article['sentences'][1]))))
            index += len(article['sentences'][1])
            if two2two:
                ret.extend(list(range(index, index+len(article['sentences'][1])-1)))
            index += len(article['sentences'][1])-1
    return ret

def getPositives(annotatedData, titles):
    pairs = []

    index = 0
    for title in sorted(titles):
        print(title)
        matches = annotatedData['matches'][titles[title]]
        for i1 in range(len(matches)):
            for i2 in range(len(matches[i1])):
                if int(matches[i1][i2]):
                    print(i1,i2)
                    if i1 >= len(annotatedData['articles'][0][titles[title]]):
                        offset = len(annotatedData['articles'][0][titles[title]])
                        sent1 = annotatedData['articles'][0][titles[title]][i1-offset]
                        sent1 += annotatedData['articles'][0][titles[title]][i1-offset+1]
                        index1 = (i1-offset,i1-offset+1)
                    else:
                        sent1 = annotatedData['articles'][0][titles[title]][i1]
                        index1 = i1
                        
                    if i2 >= len(annotatedData['articles'][1][titles[title]]):
                        offset = len(annotatedData['articles'][1][titles[title]])
                        sent2 = annotatedData['articles'][1][titles[title]][i2-offset]
                        sent2 += annotatedData['articles'][1][titles[title]][i2-offset+1]
                        index2 = (i2-offset,i2-offset+1)
                    else:
                        sent2 = annotatedData['articles'][1][titles[title]][i2]
                        index2 = i2
                        
                    pairs.append((index, index1, index2, sent1, sent2))
                index += 1
    return pairs

def getStarts(annotatedData, titles):
    ret = [0]

    for title in sorted(titles):
        print(title)
        len0 = len(annotatedData['articles'][0][titles[title]])
        len1 = len(annotatedData['articles'][1][titles[title]])
        ret.append(ret[-1] + (2*len0-1)*(2*len1-1))
    return ret

if __name__ == '__main__':
    import sys
    import gzip
    import json
    import os
    
    from altlex.embeddings import sentenceRepresentation
    
    parsedData = sys.argv[1]
    annotatedData = sys.argv[2]
    wikiFilename = sys.argv[3]
    modelFilename = sys.argv[4]

    wn = WikNetMatch()
    sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiFilename,
                                                                    modelFilename)

    titles = sentRep.titles
    titles = dict(zip(titles, range(len(titles))))
    
    with gzip.open(parsedData) as f:
        p = json.load(f)

    with gzip.open(annotatedData) as f:
        a = json.load(f)

    y = getLabels(a, getTitles(a))
    print(len(y))
    
    if os.path.exists('/tmp/features.npy'):
        X = np.load('/tmp/features.npy')
    else:

        feat1 = np.array(wn.predict(p, getTitles(a), multisentence=True)).reshape(len(y),1)
        feat2 = np.array(wn.predict(p, getTitles(a), multisentence=True, reverse=True)).reshape(len(y),1)
        feat3 = np.array(wn.predict(p, getTitles(a), multisentence=True, penalty='only')).reshape(len(y),1)
        feat4 = np.array(wn.predict(p, getTitles(a), multisentence=True, reverse=True, penalty='only')).reshape(len(y),1)

        feat5 = np.array(getLengthDiffs(p, getTitles(a), multisentence=True)).reshape(len(y),1)
        feat6 = feat5**2

        X = []
        for article in p:
            index = titles[article['title']]
            print(article['title'], index)
            embeddings = sentRep.getSentenceEmbeddingPairs(index, True)
            for simple_embedding in embeddings[1]:
                for english_embedding in embeddings[0]:
                    add = np.add(simple_embedding,english_embedding)
                    diff = abs(np.subtract(simple_embedding,english_embedding))
                    X.append(add.tolist() + diff.tolist())

        X = np.hstack((np.array(X),
                       
                       feat1,
                       feat2,
                       feat3,
                       feat4,
                       feat5,
                       feat6))
        np.save('/tmp/features.npy', X)
        
    print(X.shape)
    indices = getIndices(p, one2two=True, two2one=True)
    X = X[indices]
    y = np.array(y)[indices]
    print(X.shape, y.shape)
    
    class IdentityTransformer:
        def transform(self, X):
            return X

    classifier = Sklearner(SGDClassifier(fit_intercept=True), IdentityTransformer())
    #X = preprocessing.scale(X)
    X -= X.mean(axis=0)
    print('training')
    #for features in ((-6, -4), (-5, -3), (-6, -4, -5, -3), (-6, -4, -5, -3, -2), list(range(400)),
    #                 list(range(400)) + [-6, -4, -5, -3]):
    for features in ((list(range(400,600)) + [-6, -4, -5, -3]),):
        X_s = X[:, features]
        classifier.crossvalidate(X_s, y, printResults=True, n_folds=4) #, balanced=False)
        #print(classifier.model.coef_, classifier.model.intercept_)
        if len(features) > 10:
            features = features[:5] + features[-5:]
        np.savez('centered_'+('{}_'*len(features)).format(*features),
                 coef=classifier.model.coef_,
                 intercept=classifier.model.intercept_)
