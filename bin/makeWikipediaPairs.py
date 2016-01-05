from __future__ import print_function

import sys
import collections
import logging
import os
import gzip
import json

import nltk
import numpy as np
from scipy.stats import poisson
from sklearn.metrics.pairwise import pairwise_distances, linear_kernel

#iterate over pairs of articles
#for each article:
#create multi-sentences
#get the representation for each multi-sentence
#calculate all pairwise distances
#find the maximum (bipartite?) matching
#greedy algorithm should work
#end for

from chnlp.utils.readers import wikipediaReader
from chnlp.word2vec import sentenceRepresentation
#from wikipedia import printPairs,getMaximalMatching
from chnlp.misc import wikipedia

logger = logging.getLogger(__file__)

def calcPoissonLength(x, y, ratio=1.):
    return poisson.pmf(x[0], y[0]*ratio)

def calcGaussianLength(x, y, mean=0., std=1.):
    return norm.pdf(np.log(x[0]/y[0]), loc=mean, scale=std)

def squaredPenalty(x, y):
    return (x[0]-y[0])**2

def getMatches(title, reader, metric='cosine', n_jobs=1, lam1=.01, lam2=1):
    pairs = reader.getSentencePairs(title, pairs=True)
    numSentences = ((len(pairs[0])+1) // 2,
                    (len(pairs[1])+1) // 2)
    lengths = ([len(nltk.word_tokenize(i)) for i in pairs[0]],
               [len(nltk.word_tokenize(i)) for i in pairs[1]])
    #print(len(pairs[0]), len(pairs[1]))
    vecs = reader.getSentenceEmbeddingPairs(title, pairs=True)
    #print(len(vecs[0]), len(vecs[1]))
    
    X = np.asarray(vecs[0])
    Y = np.asarray(vecs[1])
    if X.shape[0] < 1 or Y.shape[0] < 1 or X.shape[1] != Y.shape[1]:
        print('Problem: ', title, X.shape, Y.shape, file=sys.stderr)
        return pairs, {}

    '''
    sums = [sum(lengths[0][:numSentences[0]]), sum(lengths[1][:numSentences[1]])]
    means = 1.*np.array(sums)/np.array(numSentences)
    ratio = means[1]/means[0]

    print(sums, numSentences, means, ratio)
    gram = np.exp(pairwise_distances(X,
                                     Y,
                                     metric=np.dot,
                                     n_jobs=n_jobs))
    lengthProbs = pairwise_distances(np.asarray(lengths[0]).reshape(len(lengths[0]), 1),
                                     np.asarray(lengths[1]).reshape(len(lengths[1]), 1),
                                     metric=calcPoissonLength,
                                     n_jobs=n_jobs,
                                     )#ratio=ratio)
                                     
    #create conditional probability distribution
    gram /= np.sum(gram, axis=1)[:, None]
    print(gram)
    print(lengthProbs)
    #create joint conditional probability distribution
    m = np.log(gram)+np.log(lengthProbs)
    print(m)
    '''
    m = 1-pairwise_distances(X,
                             Y,
                             metric=metric,
                             n_jobs=n_jobs)
    '''
    lengthPenalty = lam1*pairwise_distances(np.asarray(lengths[0]).reshape(X.shape[0], 1),
                                            np.asarray(lengths[1]).reshape(Y.shape[0], 1),
                                            metric=squaredPenalty,
                                            n_jobs=n_jobs)

    distancePenalty = lam2*pairwise_distances(1.*np.arange(X.shape[0]).reshape(X.shape[0], 1)/X.shape[0],
                                              1.*np.arange(Y.shape[0]).reshape(Y.shape[0], 1)/Y.shape[0],
                                              metric=squaredPenalty,
                                              n_jobs=n_jobs)
    '''
    print(m.shape)
    m = m #- lengthPenalty# - distancePenalty
    scores = {(i,j):m[i,j] for i in range(m.shape[0]) for j in range(m.shape[1])}
    #print('Scores: ', len(scores))

    #return pairs, wikipedia.getMaximalDPMatching(pairs[0][:numSentences[0]],
    #                                           pairs[1][:numSentences[1]], m)
    return pairs, wikipedia.getMaximalMatching(pairs[0][:numSentences[0]],
                                               pairs[1][:numSentences[1]],
                                               scores)

def getGreedyMatches(title, reader):
    pairs = reader.getSentencePairs(title)
    vecs = reader.getSentenceRepresentations(title)
    
    #make the shorter text the 0th axis (fewer num sentences)
    indices,values = zip(*sorted(enumerate(vecs), key=lambda x: len(x[1])))
    m = 1-pairwise_distances(np.asarray(values[0]),
                             np.asarray(values[1]),
                             metric=metric,
                             n_jobs=n_jobs)

    #find the maximum scores for all sentences
    sa = np.argsort(m, axis=None)[::-1]

    #indices = np.argpartition(m, -len(vec), axis=1)[:, -len(vec):]
    #m[np.arange(m.shape[0])[:, None], indices]

    #remove these according to a greedy matching algorithm (submodular function)
    #combine any sentences that occur consecutively
    matches = {}
    matched = set()
    for i in sa:
        i0,i1 = sa[i]/m.shape[1], sa[i]%m.shape[1]

        #if these are already a match
        if (i0,i1) in matches:
            #only allow if the second sentence is consecutive to its match
            if (i0,i1-1) not in matches and (i0,i1+1) not in matches:
                continue

            #or if the first sentence is consecutive
            if (i0-1,i1) not in matches and (i0+1,i1) not in matches:
                continue

        matches.add((i0,i1))
        matched.add(i0)
        
        #if every sentence from the shorter article has been matched, we are done
        if len(matched) >= m.shape[0]:
            break

    #first add all matches from i0_single->i1_many (many where n >= 1)
    for i0 in matches[0]:
        for i1 in sorted(matches[i0]):
            if i1 in matches[1]:
                pass
            
    return matches[0]
    

if __name__ == '__main__':
    if False:
        #logging.basicConfig(level=logging.DEBUG)
        modelFilename = sys.argv[1]
        wikiFilename = sys.argv[2]
        metric = 'cosine'
        n_jobs = 12
        sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiFilename,
                                                                        modelFilename)

        '''
        if int(sys.argv[3]) == 1:
            print('Doc2Vec')
            sentRep = sentenceRepresentation.Doc2VecEmbeddings(wikiFilename, modelFilename)
        else:
            print('WTMF')
            sentRep = sentenceRepresentation.WTMFEmbeddings(wikiFilename, modelFilename)
        '''

        for index,title in enumerate(sentRep.iterTitles()):
            print('Title: ', title.encode('utf-8'))
            pairs, matches = getMatches(index, sentRep, metric, n_jobs)
            #print('Matches: ', len(matches))
            wikipedia.printPairs(pairs[0], pairs[1], matches, 10000)
            #for match,score in matches.iteritems():
            #    print(match[0], match[1], score)
    else:
        from chnlp.misc import wiknet
        thresh = .69
        penalty = .75
        indir = sys.argv[1]
        outfile = sys.argv[2]

        wn = wiknet.WikNetMatch()

        for filename in sorted(os.listdir(indir)):
            print(filename)
            if not filename.endswith('.gz'):
                continue
            with gzip.open(os.path.join(indir, filename)) as f:
                p = json.load(f)
            for article in p:
                print(article['title'].encode('utf-8'))
                newSentences = [article['sentences'][1],
                                article['sentences'][0]]
                if not len(newSentences[0]) or not len(newSentences[1]):
                    continue
                article['sentences'] = newSentences
                scores = wn.predict([article],
                                    {article['title']:0},
                                    multisentence=True,
                                    penalty=penalty)
                scores = np.array(scores).reshape(2*len(newSentences[0])-1,
                                                  2*len(newSentences[1])-1)
                matches = np.argwhere(scores > thresh)
                with open(outfile, 'a') as f:
                    for match in matches:
                        if match[0] >= len(newSentences[0]) and match[1] >= len(newSentences[1]):
                            continue
                        if match[0] < len(newSentences[0]):
                            sent0 = wiknet.getLemmas(newSentences[0][match[0]]['words'])
                        else:
                            adjustedIndex = match[0] - len(newSentences[0])
                            sent0 = wiknet.getLemmas(newSentences[0][adjustedIndex]['words'] +
                                                     newSentences[0][adjustedIndex+1]['words'])
                        if match[1] < len(newSentences[1]):
                            sent1 = wiknet.getLemmas(newSentences[1][match[1]]['words'])
                        else:
                            adjustedIndex = match[1] - len(newSentences[1])
                            sent1 = wiknet.getLemmas(newSentences[1][adjustedIndex]['words'] +
                                                     newSentences[1][adjustedIndex+1]['words'])
                            
                        print('{}\t{}\t{}'.format(' '.join(sent0).encode('utf-8'),
                                                  ' '.join(sent1).encode('utf-8'),
                                                  scores[match[0]][match[1]]),
                              file=f)

