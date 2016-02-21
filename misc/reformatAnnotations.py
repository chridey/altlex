import os
import collections
import json
import sys
import gzip
import re

import numpy as np

from nltk import Tree
from nltk.stem import SnowballStemmer,PorterStemmer

from chnlp.utils import treeUtils
from chnlp.utils import wordUtils

from chnlp.misc import wiknet

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

def getAnnotatedFileData(f):
    ret = {'sentences': {}}
    for line in f:
        if line.startswith('Title: '):
            ret['title'] = line.replace('Title: ', '').strip()
        elif ':' in line and line[0].isdigit():
            sentenceIndex, sentence = line.strip().split(':', 1)
            ret['sentences'][int(sentenceIndex)] = sentence
    return ret

def getCausalParses(indir, parsedDir):
    titleLookup = {}
    indexLookup = {}
    for filename in os.listdir(indir):
        with open(os.path.join(indir, filename)) as f:
            data = getAnnotatedFileData(f)
            if data['title'] not in titleLookup:
                titleLookup[data['title']] = len(titleLookup)
                indexLookup[data['title']] = int(filename.replace('.0', '').replace('.1', ''))

    print(titleLookup)
    print(indexLookup)
    parsedCausalData = getParsedAnnotations({'titles': sorted(titleLookup, key=lambda x:titleLookup[x]),
                                             'indices': list(zip(*sorted(indexLookup.items(), key=lambda x:x[0])))[1]},
                                            parsedDir)

    return parsedCausalData

def labelAltlexes(connectives, goldAltlexes, foundAltlexes):
    data = []

    for connective in connectives:
        #make sure this does not overlap with any previously found altlexes
        #if len(set(range(connective[0], connective[1])) & set(foundAltlexes)):
        #    continue
            
        for i in range(connective[0], connective[1]):
            if i in goldAltlexes:
                tag = 'causal'
                classes = goldAltlexes[i][-1]
                foundAltlexes.add(goldAltlexes[i])
                break
            else:
                tag = 'notcausal'
                classes = ('Other',)

        #for i in range(connective[0], connective[1]):
        #    foundAltlexes[i] = (connective[0], connective[1], classes)

        data.append((connective, tag, classes))

    return data

def makeDataPoint(labeledDatum, parse, metadata, known=False, discovered=False):
    connective, tag, classes = labeledDatum
    phrase = parse.leaves()[connective[0]:connective[1]]

    #every connective is a new data point (although we can also limit it to the known altlexes later)
    start = treeUtils.findPhrase(phrase, metadata['words'])
                                     
    if start is None or start == 0:
        print('Problem with finding {} phrase {} in words {}'.format(tag,
                                                                     phrase,
                                                                     metadata['words']))
        return None
                        
    sentenceInfo = []
    for indices in ((start, len(metadata['words'])),
                    (0, start)):
        sentenceInfo.append({j:metadata[j][indices[0]:indices[1]] for j in metadata})

    datapoint = {'sentences': sentenceInfo,
                 'tag': tag,
                 'classes': classes,
                 'altlexLength': connective[1]-connective[0],
                 'parse': str(parse),
                 'knownAltlex': known,
                 'discoveredAltlex': discovered}

    return datapoint

altlexRegex = re.compile('##.+?##')
def extractAltlexes(indir):
    markedData = {}
    totalAltlexes = 0
    #first go through annotated files and add to lookup table
    for filename in os.listdir(indir):
        print(filename)
        if not filename.endswith('0') and not filename.endswith('1'):
            continue
        articleIndex, wikiIndex = filename.split('.')
        with open(os.path.join(indir, filename)) as f:
            data = getAnnotatedFileData(f)

        sentenceLookup = {}
        for index in data['sentences']:
            line = data['sentences'][index]
            altlexes = []
            for altlex in altlexRegex.findall(line):
                print(altlex)
                if '--' in altlex:
                    altlex,category = altlex.replace('##', '').split('--')
                else:
                    altlex,category = altlex.replace('##', '').split('++')
                    
                altlex = altlex.split()
                category = category.strip()
                if category == 'reason':
                    classes = ('Contingency.Cause.Reason',)
                elif category in ('result', 'restul'):
                    classes = ('Contingency.Cause.Result',)
                else:
                    classes = tuple()
                    print(altlex,category)
                altlexes.append((altlex,classes))
            sentenceLookup[index] = altlexes
            totalAltlexes += len(altlexes)
        markedData[(data['title'],int(wikiIndex))] = sentenceLookup

    return markedData,totalAltlexes

def getCausalAnnotations(indir, parsedData, knownAltlexes=None):
    #read in each file and return only the articleIndex, wikiIndex, sentenceIndex, altlex, and label
    #output in standard format for evaluateAltlexes.py
    stemmer = SnowballStemmer('english') #PorterStemmer()

    markedData,totalAltlexes = extractAltlexes(indir)

    dataset = []
    unfoundAltlexes = 0
    #now go through parsedData and determine every possible altlex
    for articleIndex,articleGroup in enumerate(parsedData):
        title = articleGroup['title']
        print(articleIndex, title)
        for wikiIndex,article in enumerate(articleGroup['sentences'][::-1]):
            print(wikiIndex)
            for sentenceIndex,sentenceMetadata in enumerate(article):
                metadata = {i:wiknet.getLemmas(sentenceMetadata[i]) for i in ('words',
                                                                              'lemmas',
                                                                              'pos')}
                try:
                    metadata['stems'] = [stemmer.stem(j.decode('utf-8')).lower() for j in metadata['lemmas']]
                except UnicodeEncodeError:
                    metadata['stems'] = [stemmer.stem(wordUtils.replaceNonAscii(j)).lower() for j in metadata['lemmas']]

                parse = Tree.fromstring('()')
                for parseString in sentenceMetadata['parse']:
                    try:
                        p = Tree.fromstring(parseString)
                    except ValueError:
                        print('ERROR with parse {}'.format(parseString))
                        continue
                    parse += p
                if not len(parse.leaves()):
                    continue
                
                goldAltlexes = {}
                if (title,wikiIndex) in markedData and sentenceIndex in markedData[(title,wikiIndex)]:
                    for phrase,label in markedData[(title,wikiIndex)][sentenceIndex]:
                        start = treeUtils.findPhrase(phrase, parse.leaves())
                        if start is None:
                            print('Problem with finding phrase {} in lemmas {}'.format(phrase, parse.leaves()))
                        else:
                            for i in range(start, start+len(phrase)):
                                goldAltlexes[i] = (start, start+len(phrase), label)

                foundAltlexes = set()
                #go through and find matches for any known altlexes
                leaves,pos = zip(*(parse.pos()))
                lemmas = wordUtils.lemmatize(leaves, pos)
                lemmasLower = [i.lower() for i in lemmas]
                knownConnectives = []
                if knownAltlexes is not None:
                    foundIndices = set()
                    #go from the longest length to the shortest
                    for length in range(len(lemmasLower))[::-1]:
                        for i in range(len(lemmasLower)-length):
                            j = i+length
                            l = tuple(lemmasLower[i:j]) + tuple(pos[i:j])
                            if i not in foundIndices and j not in foundIndices:
                                if l in knownAltlexes:
                                    knownConnectives.append((i,j))
                                    foundIndices.update(set(range(i,j)))

                    '''
                    for i in range(len(lemmasLower)):
                        for j in range(i+1, len(lemmasLower)):
                            l = tuple(lemmasLower[i:j]) + tuple(pos[i:j])
                            #print(l, l in knownAltlexes)
                            if l in knownAltlexes:
                                knownConnectives.append((i,j))
                    '''
                #if len(knownConnectives):
                #    print('knownConnectives: ', knownConnectives)
                
                #go through and try to discover any other altlexes
                '''
                discoveredConnectives = treeUtils.getConnectives(parse,
                                                                 blacklist = {tuple(k.split()) for k in wordUtils.modal_auxiliary},
                                                                 whitelist = wordUtils.all_markers,
                                                                 leaves = lemmas,
                                                                 verbose=False)
                '''
                discoveredConnectives = []

                #print(lemmas)
                inBothConnectives = set(knownConnectives) & set(discoveredConnectives)
                for labeledDatum in labelAltlexes(inBothConnectives, goldAltlexes, foundAltlexes):
                    dataset.append(makeDataPoint(labeledDatum, parse, metadata, True, True))

                onlyKnownConnectives = set(knownConnectives) - set(discoveredConnectives)
                for labeledDatum in labelAltlexes(onlyKnownConnectives, goldAltlexes, foundAltlexes):
                    datum = makeDataPoint(labeledDatum, parse, metadata, True)
                    #if datum is not None:
                    #    print(labeledDatum, datum['sentences'][0]['lemmas'], datum['altlexLength'])
                    dataset.append(datum)

                onlyDiscoveredConnectives = set(discoveredConnectives) - set(knownConnectives)
                for labeledDatum in labelAltlexes(onlyDiscoveredConnectives, goldAltlexes, foundAltlexes):
                    dataset.append(makeDataPoint(labeledDatum, parse, metadata, False, True))
                
                for altlex in set(goldAltlexes.values()) - foundAltlexes:
                    unfoundAltlexes += 1
                    print('altlex {} not found in {}:{}'.format(altlex,
                                                                metadata['words'],
                                                                parse))

        #TODO: multisentence

    print("Total: {} Unfound: {}".format(totalAltlexes, unfoundAltlexes))
    print("Total Positive Training: {} Total Negative Training: {}".format(sum(datapoint['tag']=='causal' for datapoint in filter(None,dataset)),
                                                                           sum(datapoint is not None for datapoint in dataset)))
    print("Total Unique Altlexes: {}".format(len(set(tuple(datapoint['sentences'][0]['words'][:datapoint['altlexLength']]) for datapoint in filter(None,dataset) if datapoint['tag']=='causal'))))
    return list(filter(None, dataset))
                
        
def getParsedAnnotations(annotatedData, indir):
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
        found = set(indices.keys()) & set(range(int(start), int(end)+1))
        if len(found):
            with gzip.open(os.path.join(indir, filename)) as f:
                p = json.load(f)
            for article in p:
                if article['title'] in titles:
                    print(article['title'])
                    newSentences = [article['sentences'][1],
                                    article['sentences'][0]]
                    article['sentences'] = newSentences
                    ret.append(article)
                    foundTitles.add(article['title'])
        if len(foundTitles) >= len(titles):
            break

    print(foundTitles)
    print(len(foundTitles))
    print(len(titles))
    assert(foundTitles == set(titles))
    parses = sorted(ret, key=lambda x:x['title'])
    return parses

def calculateKrippendorff(dirs, origDir):
    #each directory should have the same files in it
    #first read through each file and mark the file and index where an altlex is marked
    #then go through one file

    sentenceLookup = collections.defaultdict(dict)
    users = set(dirs)
    for indir in dirs:
        print(indir)
        markedData,totalAltlexes = extractAltlexes(indir)
        for title,wikiIndex in markedData:
            for sentenceIndex in markedData[(title,wikiIndex)]:
                if not len(markedData[(title,wikiIndex)][sentenceIndex]):
                    continue
                print(title,wikiIndex,sentenceIndex)
                sentenceLookup[(title,wikiIndex,sentenceIndex)][indir] = markedData[(title,wikiIndex)][sentenceIndex]
                
    #need to determine for each sentence if they found the same phrases
    ret = []
    for filename in os.listdir(origDir):
        if not filename.endswith('0') and not filename.endswith('1'):
            continue
        print(filename)
        articleIndex, wikiIndex = filename.split('.')
        with open(os.path.join(origDir, filename)) as f:
            data = getAnnotatedFileData(f)

        for sentenceIndex in data['sentences']:
            if (data['title'],int(wikiIndex),sentenceIndex) in sentenceLookup:
                print (sentenceLookup[(data['title'],int(wikiIndex),sentenceIndex)])
                line = data['sentences'][sentenceIndex].strip().split()
                #for each altlex, find it in the text and add an index lookup
                indices = collections.defaultdict(dict)
                for user in sentenceLookup[(data['title'],int(wikiIndex),sentenceIndex)].keys():
                    for altlex,classes in sentenceLookup[(data['title'],int(wikiIndex),sentenceIndex)][user]:
                        i = 0
                        while (i < len(line)):
                            print(altlex, line[i:i+len(altlex)])
                            if line[i:i+len(altlex)] == altlex:
                                indices[i][user] = len(altlex),classes[0]
                                i += len(altlex)
                            else:
                                i += 1
                print(indices)
                #now go through each word in the sentence and determine if there is an altlex at that location, accounting for overlap
                i = 0
                m = 0
                start = None
                altlexIndices = collections.defaultdict(dict)
                while i < len(line):
                    print(i,m)
                    if m == 1:
                        m = 0
                    elif m:
                        m -= 1
                        altlexIndices[start].update(indices[start])
                        
                    if i in indices:
                        m += max(l[0] for l in indices[i].values())
                        start = i
                        altlexIndices[start].update(indices[i])
                    i += 1

                for i in altlexIndices:
                    for user in altlexIndices[i]:
                        ret.append((user,
                                    '{}.{}.{}.{}'.format(data['title'],
                                                         wikiIndex,
                                                         sentenceIndex,
                                                         i),
                                    indices[i][user][1]))
                    for user in users-set(altlexIndices[i]):
                        ret.append((user,
                                    '{}.{}.{}.{}'.format(data['title'],
                                                         wikiIndex,
                                                         sentenceIndex,
                                                         i),
                                    'Other'))
                        
    return ret

if __name__ == '__main__':

    if sys.argv[3] == '0':
        reformatWashington(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == '1':
        reformatAnnotations(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == '2':
        with gzip.open(sys.argv[4]) as f:
            a = json.load(f)
        parses = getParsedAnnotations(a, sys.argv[1])
        with gzip.open(sys.argv[2], 'w') as f:
            json.dump(parses, f)
    elif sys.argv[4] == '3':
        parses = getCausalParses(sys.argv[1], sys.argv[2])
        with gzip.open(sys.argv[3], 'w') as f:
            json.dump(parses, f)
    else:
        with gzip.open(sys.argv[1]) as f:
            parses = json.load(f)
        if len(sys.argv) > 5:
            with open(sys.argv[5]) as f:
                altlexes = {tuple(i.split()) for i in f.read().splitlines()}
        else:
            altlexes = None
        data = getCausalAnnotations(sys.argv[2], parses, altlexes)
        with gzip.open(sys.argv[3], 'w') as f:
            json.dump(data, f)
