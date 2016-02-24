from __future__ import print_function

import sys
import collections
import logging
import os
import gzip
import json
import time

import nltk
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.externals import joblib

from altlex.embeddings import sentenceRepresentation
from altlex.wiknet import wiknet
from altlex.wiknet import parallelWikipedia
        
logger = logging.getLogger(__file__)

#batch wiknetmatch for parallelization
def wiknetBatch(batchIndex, batch, penalty=0, wn=None):
    if wn is None:
        with gzip.open('/local/nlp/chidey/simplification/scores.json.gz') as f:
            s = json.load(f)
        wn = wiknet.WikNetMatch(scores=s)
    scores = []
    for a,b in batch:
        if a is None or b is None:
            if penalty == 'both':
                scores.append((0.,0.))
            else:
                scores.append(0)
        else:
            scores.append(wn(a, b, penalty=penalty))
    return batchIndex,scores

def getFilenames(indir):
    filenameLookup = {}
    for filename in parallelWikipedia.iterFilenames(indir):
        start,end,junk = filename.split('.', 2)
        start = int(start)
        end = int(end)
        for index in range(start, end-1):
            filenameLookup[index] = filename,start
    return filenameLookup
            
if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)
    modelFilename = sys.argv[1]
    wikiFilename = sys.argv[2]
    indir = sys.argv[3]
    outfile = sys.argv[4]

    if len(sys.argv) > 5:
        n_jobs = int(sys.argv[5])
    else:
        n_jobs = 1

    if len(sys.argv) > 6:
        startArticle = int(sys.argv[6])
    else:
        startArticle = 0

    #get the embedding similarities as well as the wiknet similarities and unmatched words        
    metric = 'cosine'
    sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiFilename,
                                                                    modelFilename)
    wn = wiknet.WikNetMatch('/local/nlp/chidey/simplification/WikNet_word_pairs')
    thresh = .5
    penalty = 'both'

    parallel = joblib.Parallel(n_jobs=n_jobs)

    titles = sentRep.titles
    filenameLookup = getFilenames(indir)
    filename = None
    for i in range(startArticle, len(titles)):
        print(i, filenameLookup.get(i, None))
        if i not in filenameLookup:
            #poll until this file is created by the parser
            while 1:
                filenameLookup = getFilenames(indir)
                if i in filenameLookup:
                    break
        if filenameLookup[i][0] != filename:
            print('opening file')
            filename = filenameLookup[i][0]
            start = filenameLookup[i][1]
            with gzip.open(os.path.join(indir, filename)) as f:
                p = json.load(f)

        if filename is not None:
            article = p[i-start]
            try:
                print(article['title'])
            except UnicodeEncodeError:
                print('UNICODE: ', article['index'])
                
            sentences = article['sentences']

            if sentences[0] is None or sentences[1] is None or not len(sentences[0]) or not len(sentences[1]):
                continue

            startTime = time.time()
            embedding_scores = sentRep.batchSimilarity(article['index'],
                                                       pairs=True,
                                                       n_jobs=n_jobs)
            print(time.time() - startTime)
            total_length = len(embedding_scores)
            print(total_length, 2*len(sentences[0])-1, 2*len(sentences[1])-1)
            if total_length != (2*len(sentences[0])-1)*(2*len(sentences[1])-1):
                print('length mismatch at {}'.format(article['index']), file=sys.stderr)
                continue
            embedding_scores = np.array(embedding_scores).reshape(2*len(sentences[0])-1,
                                                                  2*len(sentences[1])-1)

            #only use multiple cores if the job is expected to take longer than 4 seconds

            startTime = time.time()
            if total_length > 10000:
                batch_length = total_length // n_jobs
                
                results = parallel(joblib.delayed(wiknetBatch)(batchIndex,
                                                               batch,
                                                               penalty) for batchIndex, batch in parallelWikipedia.iterBatchedArticlePairs(sentences, batch_length, multisentence=True))
                wiknet_scores = []
                for result in sorted(results, key=lambda x:x[0]):
                    wiknet_scores.extend(result[1])
            else:
                batch = list(parallelWikipedia.iterArticlePairs(sentences, multisentence=True))
                wiknet_scores = wiknetBatch(0, batch, penalty, wn)[1]
            print(time.time()-startTime)
            if penalty == 'both':
                wiknet_scores, wiknet_penalties = zip(*wiknet_scores)
            wiknet_penalties = np.array(wiknet_penalties, dtype=np.object).reshape(2*len(sentences[0])-1,
                                                                                   2*len(sentences[1])-1)
            
            wiknet_scores = np.array(wiknet_scores, dtype=np.object).reshape(2*len(sentences[0])-1,
                                                                             2*len(sentences[1])-1)

            with open(outfile, 'a') as f:
                for i in range(embedding_scores.shape[0]):
                    if i >= len(sentences[0]):
                        m = len(sentences[1])
                        index1 = i - len(sentences[0]), i - len(sentences[0]) + 1
                    else:
                        m = embedding_scores.shape[1]
                        index1 = i
                    for j in range(m):
                        escore = embedding_scores[i,j]
                        wscore = wiknet_scores[i,j]
                        pen = wiknet_penalties[i,j]
                        
                        if escore >= thresh or wscore >= thresh:
                            if j >= len(sentences[1]):
                                index2 = j - len(sentences[1]), j - len(sentences[1]) + 1
                            else:
                                index2 = j

                            print(article['index'], index1, index2, escore, wscore, pen,
                                  sep='\t',
                                  file=f)

