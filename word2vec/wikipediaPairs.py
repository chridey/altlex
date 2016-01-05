import itertools

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def getSentenceVectors(model, fileIndex, articleIndex):
    sentenceVectors = [model.docvecs['SENT_{}_{}_0_0'.format(fileIndex,articleIndex)]]
    pairVectors = [model.docvecs['SENT_{}_{}_1_1'.format(fileIndex,articleIndex)]]
    for s in itertools.count(2):
        if 'SENT_{}_{}_{}_1'.format(fileIndex,articleIndex,s) not in model.docvecs:
            break
        sentenceVectors.append(model.docvecs['SENT_{}_{}_{}_0'.format(fileIndex,articleIndex,s)])
        pairVectors.append(model.docvecs['SENT_{}_{}_{}_1'.format(fileIndex,articleIndex,s)])
    sentenceVectors.append(model.docvecs['SENT_{}_{}_{}_0'.format(fileIndex,articleIndex,s)])

    return sentenceVectors + pairVectors

def getSimMatrix(v1, v2):
    return 1-pairwise_distances(np.asarray(v1),
                                np.asarray(v2),
                                metric='cosine',
                                n_jobs=12)

def getMatching(set1, set2, s, k=10):
    sa = np.argsort(s, axis=None)
    #ss = np.sort(s, axis=None)
    pairs = list(zip(sa[-k:]/s.shape[1], sa[-k:]%s.shape[1]))[::-1]
    eset1 = set1 + list(map(' '.join, zip(set1, set1[1:])))
    eset2 = set2 + list(map(' '.join, zip(set2, set2[1:])))
    return [(eset1[i1],eset2[i2],s[i1][i2]) for i1,i2 in pairs]

def getParallelArticles(title, discourse=None, source=None):
    assert(source is not None)
    corpora = []
    for i in range(len(source.filenames)):
        j = source.articles[i].index(title)
        sentences = source.getArticle(i, j)
        #sentences += list(map(' '.join, zip(sentences, sentences[1:])))
        corpora.append(sentences)
    return corpora

def getSimilarityScores(title, model, source):
    m = []
    for i in range(len(source.filenames)):
        j = source.getArticleIndex(i, title)
        m.append(getSentenceVectors(model, i, j))
    return getSimMatrix(m[0], m[1])
