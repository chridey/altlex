import os
import collections
import json
import sys
import gzip

import numpy as np

def reformatWashington(infile, outfile):
    titleLookup = {}
    articles = [[], []]
    matches = []
    with open(infile) as f:
        for line in f:
            title, sentence1, sentence2, rating = line.split('\t')
            if title not in titleLookup:
                titleLookup[title] = len(titleLookup)
                sentenceLookup = [{}, {}]
                articles[0].append([])
                articles[1].append([])
                matches.append(collections.defaultdict(dict))
            if sentence1 not in sentenceLookup[0]:
                articles[0][titleLookup[title]].append(sentence1)
                sentenceLookup[0][sentence1] = len(sentenceLookup[0])
            if sentence2 not in sentenceLookup[1]:
                articles[1][titleLookup[title]].append(sentence2)
                sentenceLookup[1][sentence2] = len(sentenceLookup[1])
            matches[titleLookup[title]][sentenceLookup[0][sentence1]][sentenceLookup[1][sentence2]] = int(rating)
    with gzip.open(outfile, 'w') as f:
        json.dump({'titles': sorted(titleLookup, key=lambda x:titleLookup[x]),
                   'articles': articles,
                   'matches': matches,
                   'files': ['en', 'simple'],
                   'starts': []},
                  f)

def reformatAnnotations(indir, outfile):
    titleLookup = {}
    indexLookup = {}
    articles = [[], []]
    matches = []

    for subdir in os.listdir(indir): #('jessica', 'melody'):
        if os.path.isdir(subdir):
            fullsubdir = os.path.join(indir, subdir)
            for filename in os.listdir(fullsubdir):
                print(filename)
                if filename[0].isdigit() and filename.endswith('.simple') or filename.endswith('.en'):
                    fileIndex = 1-filename.endswith('.simple')
                    with open(os.path.join(fullsubdir, filename)) as f:
                        for lineNum,line in enumerate(f):
                            if line.startswith('Title:'):
                                title = line.replace('Title: ', '').strip()
                                if title not in titleLookup:
                                    titleLookup[title] = len(titleLookup)
                                    indexLookup[title] = int(filename.replace('.simple', '').replace('.en', ''))
                                    articles[0].append([])
                                    articles[1].append([])
                                    matches.append(collections.defaultdict(dict))
                                    
                            elif not line:
                                if len(articles[fileIndex][titleLookup[title]]) < lineNum:
                                    articles[fileIndex][titleLookup[title]].append('')
                            else:
                                ret = line.split()
                                if len(ret) < 1 or not ret[0][:-1].isdigit():
                                    if len(articles[fileIndex][titleLookup[title]]) < lineNum:
                                        articles[fileIndex][titleLookup[title]].append('')
                                    continue
                                
                                line, annot = ret[:2]
                                if not annot[0] == '#' or not annot[-1] == '#':
                                    if len(articles[fileIndex][titleLookup[title]]) < lineNum:
                                        articles[fileIndex][titleLookup[title]].append(' '.join(ret[1:]))
                                    continue
                               
                                matchingIndex, partial, disc = annot[1:-1].split('.')
                                if ',' in matchingIndex:
                                    indices = matchingIndex.split(',')
                                else:
                                    indices = [matchingIndex]
                                for index in indices:
                                    matches[titleLookup[title]][subdir][(int(line.replace(':', '')),
                                                                         int(index))] = partial
                                if len(articles[fileIndex][titleLookup[title]]) < lineNum:
                                    articles[fileIndex][titleLookup[title]].append(' '.join(ret[2:]))

    print(titleLookup)
    '''
    for wiki in articles:
        print('WIKI')
        for article in wiki:
            print(len(article))
    for match in matches:
        print(match)
    '''
    #for each simple sentence, we have the corresponding english sentence for multiple annotators
    agreedMatches = []
    for index,match in enumerate(matches):
        m = [collections.defaultdict(dict), collections.defaultdict(dict)]
        total = collections.defaultdict(int)
        for subdir in match:
            for pair in match[subdir]:
                total[pair] += 1
        numAnnotators = len(match)
        for simple,english in total.keys():
            #for each annotator, if at least half think this is a paraphrase, mark it
            if 1.*total[(simple,english)]/numAnnotators >= .5:
                m[1][english][simple] = 1
                m[0][simple][english] = 1
        agreedMatches.append(m)

    print(agreedMatches)
    
    #for each 2-to-1 or 1-to-2 match, add it unless its already added (break ties by taking the match with the closer index), TODO: mark partial as those that have remaining matches
    finalMatches = []
    for index,match in enumerate(agreedMatches):
        m = np.zeros((2*len(articles[0][index])-1, 2*len(articles[1][index])-1))
        print(index, m.shape)
        matched = [set(), set()]
        #first get the 1-to-2 matches from english to simple, then do the other way around
        for i in (1,0):
            print(i)
            for first in sorted(match[i].keys()):
                for second in sorted(match[i][first].keys()):
                    if first not in matched[i] and second not in matched[1-i] and second+1 not in matched[1-i] and second+1 in match[i][first]:
                        print(first, second)
                        if i == 1:
                            m[(len(articles[0][index])+second,first)] = 3
                        else:
                            m[(first,len(articles[1][index])+second)] = 3
                        matched[i] |= {first}
                        matched[1-i] |= {second, second+1}

        #match any remaining 1-to-1 matches that are not included in 2-to-1 or 1-to-2
        for simple in sorted(match[0].keys()):
            for english in sorted(match[0][simple].keys()):
                if english not in matched[1] and simple not in matched[0]:
                    m[(simple, english)] = 3
                    matched[0] |= {simple}
                    matched[1] |= {english}

        finalMatches.append(m.tolist())

    with gzip.open(outfile, 'w') as f:
        json.dump({'titles': sorted(titleLookup, key=lambda x:titleLookup[x]),
                   'indices': list(zip(*sorted(indexLookup.items(), key=lambda x:x[0])))[1],
                   'articles': articles,
                   'matches': finalMatches,
                   'files': ['simple', 'en'],
                   'starts': []},
                  f)
        
def getParsedAnnotations(annotatedData, indir, outfile):
    #read in the directory with the entire parsed wikipedia and output only those from the given titles
    titles = dict(zip(annotatedData['titles'], range(len(annotatedData['titles']))))
    indices = dict(zip(annotatedData['indices'], range(len(annotatedData['indices']))))

    ret = []
    foundTitles = set()
    for filename in sorted(os.listdir(indir)):
        print(filename)
        if not filename.endswith('.gz'):
            continue
        start, end, count, ext1, ext2 = filename.split('.')
        found = set(indices.keys()) & set(range(int(start), int(end)))
        if len(found):
            with gzip.open(os.path.join(indir, filename)) as f:
                p = json.load(f)
            for article in p:
                print(article['title'])
                if article['title'] in titles:
                    newSentences = [article['sentences'][1],
                                    article['sentences'][0]]
                    article['sentences'] = newSentences
                    ret.append(article)
                    foundTitles.add(article['title'])
        if len(foundTitles) >= len(titles):
            break

    print(len(foundTitles))
    print(len(titles))
    assert(foundTitles == set(titles))
    parses = sorted(ret, key=lambda x:x['title'])

    with gzip.open(outfile, 'w') as f:
        json.dump(parses, f)

if __name__ == '__main__':
    from chnlp.misc import wiknet

    if sys.argv[3] == '0':
        reformatWashington(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == '1':
        reformatAnnotations(sys.argv[1], sys.argv[2])
    else:
        with gzip.open(sys.argv[4]) as f:
            a = json.load(f)
        getParsedAnnotations(a, sys.argv[1], sys.argv[2])
