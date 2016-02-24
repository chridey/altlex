import os
import gzip
import json

from altlex.wiknet import wiknet

def iterFilenames(indir, verbose=False):
    for filename in sorted(os.listdir(indir)):
        if verbose:
            print(filename)

        if not filename.endswith('.gz'):
            continue

        yield filename
        
def iterParsedParallelWikipedia(indir, verbose=False):
    for filename in iterFilenames(indir, verbose):
        with gzip.open(os.path.join(indir, filename)) as f:
            p = json.load(f)

            for article in p:
                yield article

def iterArticlePairs(article, multisentence=False):

    for a in article[0]:
        for b in article[1]:
            yield a,b

        if multisentence:
            for i in range(len(article[1])-1):
                b = {'lemmas': article[1][i]['lemmas'] + article[1][i+1]['lemmas'],
                     'pos': article[1][i]['pos'] + article[1][i+1]['pos'],
                     'dep': article[1][i]['dep'] + article[1][i+1]['dep']}
                yield a,b
                
    if multisentence:
        for i in range(len(article[0])-1):
            a = {'lemmas': article[0][i]['lemmas'] + article[0][i+1]['lemmas'],
                 'pos': article[0][i]['pos'] + article[0][i+1]['pos'],
                 'dep': article[0][i]['dep'] + article[0][i+1]['dep']}
            for b in article[1]:
                yield a,b

            for i in range(len(article[1])-1):
                yield None,None

def iterBatchedArticlePairs(article, batchLength, multisentence=None):
    total = 0
    batchIndex = 0
    ret = []
    for pair in iterArticlePairs(article, multisentence):
        ret.append(pair)
        if len(ret) >= batchLength:
            yield batchIndex,ret
            ret = []
            batchIndex += 1
    if len(ret):
        yield batchIndex,ret
    
