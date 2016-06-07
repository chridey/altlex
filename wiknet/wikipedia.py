from __future__ import print_function

import re
import time
import itertools
import os
import shutil
import sys

import nltk

bracketRe = re.compile('(.*?)(\[[^\[]*?\]|\Z)')
stopwords = set(nltk.corpus.stopwords.words('english'))
sn = nltk.stem.SnowballStemmer('english')

def cache(func):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            ret = func(*args)
            cache[args] = ret
            return ret
    def clear():
        cache.clear()
    setattr(wrapper, 'clear', clear)
    
    return wrapper

def loadDiscourse(file, format=1):
    with open(file) as f:
        if format == 1:
            return {tuple(getUnigrams(i)) for i in f.read().splitlines()}
        elif format == 0:
            return f.read().splitlines()
        else:
            #make trie
            ret = {}
            for connective in f.read().splitlines():
                print (connective)
                words = connective.split()
                node = ret
                for word in words[:-1]:
                    if word not in node:
                        node[word] = {}
                    node = node[word]
                if words[-1] not in node:
                    node[words[-1]] = {}
                node[words[-1]][None] = None
            return ret                
        #return set(f.read().splitlines())

@cache
def getUnigrams(sentence):
    return [sn.stem(i).lower() for i in nltk.word_tokenize(sentence)]

def getBagOfNgrams(sentence, filter=True):
    if filter:
        unigrams = [(i,) if i not in stopwords and i.isalnum() else 'NULL' 
                    for i in getUnigrams(sentence)]
    else:
        unigrams = [(i,) for i in getUnigrams(sentence)]
    bigrams = set(zip(unigrams, unigrams[1:]))
    trigrams = set(zip(unigrams, unigrams[1:], unigrams[2:]))
    words = (set(unigrams) | bigrams) - {('NULL',), ('NULL', 'NULL')}

    return words

def normalizeIndex(index, s):
    if index >= len(s):
        i0 = index - len(s)
        i1 = i0 + 1
    else:
        i0 = index
        i1 = None

    return i0,i1

def getMultiSentence(s, index1, index2):
    multisentence = s[index1]
    if index2 is not None:
        multisentence = ' '.join((multisentence, s[index2]))
    return multisentence

def printPairs(set1, set2, scores, k=10, handle=sys.stdout):
    s = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    for pair,score in s[:k]:
        for index, s in ((pair[0], set1), (pair[1], set2)):
            i0, i1 = normalizeIndex(index, s)
            print(getMultiSentence(s, i0, i1).encode('utf-8'), end=' , ')

        print(score)

def getMaximalMatching(set1, set2, scores, discourse=None, minLength=4, one2twoOnly=False):
    s = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    newScores = {}
    newSet = [set(), set()]
    count = 0

    #print(len(set1), len(set2))
    for pair,score in s:
        count += 1
        i0, i1 = normalizeIndex(pair[0], set1)
        if i0 in newSet[0] or (i1 is not None and i1 in newSet[0]):
            continue

        i2, i3 = normalizeIndex(pair[1], set2)
        if i2 in newSet[1] or (i3 is not None and i3 in newSet[1]):
            continue

        #only consider pairs of 1->2 sentences of 2->1 sentences
        if one2twoOnly:
            if (i1 is None and i3 is None) or (i1 is not None and i3 is not None):
                continue

        if len(set1[i0].split()) < minLength:
            continue
        if i1 is not None and len(set1[i1].split()) < minLength:
            continue
        if len(set2[i2].split()) < minLength:
            continue
        if i3 is not None and len(set2[i3].split()) < minLength:
            continue
        
        multisentence1 = getMultiSentence(set1, i0, i1)
        multisentence2 = getMultiSentence(set2, i2, i3)

        #print(pair, len(set1), len(set2), i0, i1, i2, i3)
        if not len(multisentence1) or not len(multisentence2):
            continue
        if not((multisentence1[0].istitle() or multisentence1[0].isdigit() or multisentence1[0] in ('"', "'")) and
               multisentence1[-1] in ('.', '!', '?')):
            continue
        if not((multisentence2[0].istitle() or multisentence2[0].isdigit() or multisentence2[0] in ('"', "'")) and
               multisentence2[-1] in ('.', '!', '?')):
            continue

        if discourse is not None:
            words1 = set(getBagOfNgrams(multisentence1, filter=False))
            words2 = set(getBagOfNgrams(multisentence2, filter=False))
            if not (len(discourse & words1) or len(discourse & words2)):
                continue

        newSet[0].update({i0, i1})
        newSet[1].update({i2, i3})
        newScores[pair] = score
        if len(newSet[0]) >= len(set1)+1:
            break
        if len(newSet[1]) >= len(set2)+1:
            break
        
    return newScores

