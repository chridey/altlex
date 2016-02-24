from __future__ import print_function

import re
import time
import itertools
import os
import shutil
import sys

import nltk

import requests
import bs4

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

def loadDiscourse(file='/home/chidey/PDTB/chnlp/config/markers', format=1):
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

def splitOnDiscourse(sentence, discourse, allConnectives=False):
    unigrams = getUnigrams(sentence)
    words = {length: zip(*(unigrams[i:] for i in range(length))) for length in range(1, max(map(len,discourse))+1)}
    ngrams = set(itertools.chain(*words.values()))
    relations = ngrams & discourse

    if allConnectives:
        return relations
    
    if len(relations):        
        #if there is more than one, take the most centered one
        relation, length, distance = min(((relation, length, abs(len(words)/2.-words[length].index(relation)))
                                          for length,matches in itertools.groupby(relations, len)
                                          for relation in matches),
                                         key = lambda x:x[2])
        #print(relation, length, distance, len(words)/2.)
        index = words[length].index(relation)
        splitSentence = nltk.word_tokenize(sentence)
        return (splitSentence[:index],
                splitSentence[index:index+len(relation)],
                splitSentence[index+len(relation):])
    return None

def getParallelArticles(title, discourse=None):
    corpora = []
    for wiki in ('en', 'simple'):
        print(wiki)
        r = requests.get('https://{}.wikipedia.org/wiki/{}'.format(wiki, title))
        soup = bs4.BeautifulSoup(r.text)#, 'lxml')
        sentences = []
        for paragraph in soup.find_all('p'):
            if '.' in paragraph.text:
                m = [i for i in bracketRe.finditer(paragraph.text)]
                p = ' '.join(zip(*(i.groups() for i in m))[0])
                #sentences += nltk.sent_tokenize(p)
                #split sentences on discourse relations
                for sent in nltk.sent_tokenize(p):
                    if discourse:
                        clauses = splitOnDiscourse(sent, discourse)
                        if clauses is not None:
                            sentences.extend([' '.join(clauses[0]),
                                              ' '.join(clauses[1]+clauses[2])])
                            continue
                    sentences.append(sent)
        corpora.append(sentences)
    return corpora

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

def getMatching(set1, set2):
    #do jaccard index or whatever similarity metric

    scores = {}
    sets1 = []
    sets2 = []
    for sent1 in set1:
        sets1.append(getBagOfNgrams(sent1))
    for sent2 in set2:
        sets2.append(getBagOfNgrams(sent2))

    sets1.extend([i|j for i,j in zip(sets1, sets1[1:])])
    sets2.extend([i|j for i,j in zip(sets2, sets2[1:])])

    start1 = 0 #len(set1)
    start2 = 0 #len(set2)
    for i1, sent1 in enumerate(sets1[start1:]):
        if not len(sent1):
            continue
        for i2, sent2 in enumerate(sets2[start2:]):
            if not len(sent2):
                continue
            #for now do jaccard similarity
            scores[(i1+start1,i2+start2)] = 1.*len(sent1 & sent2)/len(sent1 | sent2)

    #for i1, sent1 in enumerate(sets1):
    #    for i2, sent2 in enumerate(sets2):
    #        scores[(i1,i2)] = 1.*len(sent1 & sent2)/len(sent1 | sent2)

    #TODO: what about consecutive sentences where one contains a discourse marker?
            
    return scores

def normalizeIndex(index, s):
    if index >= len(s):
        i0 = index - len(s)
        i1 = i0 + 1
    else:
        i0 = index
        i1 = None

    return i0,i1

def reverseIndex(i0, i1, s):
    if i1 is None:
        return i0
    return i0 + len(s)

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

def getMaximalDPMatching(set1, set2, scores):
    maxes = scores.copy() #np.zeros(shape=(len(set1), len(set2)))
    argmaxes = {}

    for combo in range(len(set1)+len(set2)-2):
        for i in range(combo+1):
            print(combo, i)
            if i >= len(set1):
                continue
            j = combo-i
            if j >= len(set2):
                continue
            if j > 0 and maxes[i,j-1] > maxes[i,j]:
                maxes[i,j] = maxes[i,j-1]
                argmaxes[(i,j)] = (i, j-1)
            elif i > 0 and maxes[i-1, j] > maxes[i,j]:
                maxes[i,j] = maxes[i-1, j]
                argmaxes[(i,j)] = (i-1, j)
            elif i > 0 and j > 0 and maxes[i-1, j-1] + scores[(i,j)] > maxes[i,j]:
                maxes[i,j] = maxes[i-1, j-1] + scores[(i,j)]
                argmaxes[(i,j)] = (i-1, j-1)
            elif i > 0 and j > 1 and maxes[i-1, j-2] + scores[(i, reverseIndex(j-1, j, set1))] > maxes[i,j]:
                maxes[i,j] = maxes[i-1, j-2] + scores[(i, reverseIndex(j-1, j, set2))]
                argmaxes[(i,j)] = (i-1, j-2)
            elif i > 1 and j > 0 and maxes[i-2, j-1] + scores[(reverseIndex(i-1, i, set1), j)] > maxes[i,j]:
                maxes[i,j] = maxes[i-2, j-1] + scores[(reverseIndex(i-1, i, set1), j)]
                argmaxes[(i,j)] = (i-2, j-1)
            print(maxes[i,j], argmaxes[(i,j)])
            
    i,j = len(set1)-1,len(set2)-1
    pairs = {}
    while i>=0 and j>=0:
        i_new, j_new = argmaxes[(i,j)]

        #check if matches with null sentence
        if i_new == i or j_new == j:
            i,j = i_new, j_new
            continue

        if i_new == i-1:
            part1 = i
        elif i_new == i-2:
            part1 = reverseIndex(i-1, i, set1)
        if j_new == j-1:
            part2 = j
        elif j_new == j-2:
            part2 = reverseIndex(j-1, j, set2)
            
        pairs[(part1, part2)] = maxes[i,j]
        i,j = i_new, j_new

    return pairs
    
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

def main(thresh=.24, interval=None, timeout=None):
    starttime = time.time()

    try:
        discourse = loadDiscourse('markers')
        causal = loadDiscourse('markers_causal')

        with open('simplewiki.titles') as f:
            titles = f.read().splitlines()

        pairs = {}
        for title in titles:
            print(title)
            en,simple = getParallelArticles(title) #, discourse)
            scores = getMatching(en, simple)
            new = getMaximalMatching(en, simple, scores, discourse) #causal)

            for pair in new:
                if new[pair] < thresh:
                    continue

                #dont care about pairs that are exactly the same .... or do I?
                if new[pair] == 1:
                    continue

                i0, i1 = normalizeIndex(pair[0], en)
                i2, i3 = normalizeIndex(pair[1], simple)
                multisentence1 = getMultiSentence(en, i0, i1)
                multisentence2 = getMultiSentence(simple, i2, i3)
                pairs[(multisentence1, multisentence2)] = new[pair]

            print(len(pairs))

            if timeout is not None and time.time()-starttime > timeout*60:
                raise Exception
            if interval is not None and time.time()-starttime > interval*60:
                writePairs(pairs)

    except Exception as e:
        raise e

    return pairs

def writePairs(pairs, outfile = 'pairs.out'):
    if os.path.exists(outfile):
        shutil.copy(outfile, outfile + '.backup')
    with open(outfile, 'w') as f:
        for pair in pairs:
            print('{} , {} , {}'.format(pair[0].encode('utf-8'), 
                                        pair[1].encode('utf-8'), 
                                        pairs[pair]), 
                  file=f)

if __name__ == '__main__':
    pairs = main(interval=1)
    writePairs(pairs)

