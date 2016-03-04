import sys
import gzip
import json
import os
import collections

from altlex.wiknet import parallelWikipedia

def makeParsedPair(index1, index2, article, lookupSet):
    pair = [None, None]
    for wikiIndex,index in ((0,index1), (1,index2)):
        for sentenceIndex in index:
            sentence = article['sentences'][wikiIndex][sentenceIndex]
            lemmas = []
            for lemmas_partial in sentence['lemmas']:
                lemmas += [x.lower() for x in lemmas_partial]
            words = filter(lambda x:x.isalpha(), lemmas)
            if tuple(sorted(words)) in lookupSet:
                return None
            if pair[wikiIndex] is None:
                pair[wikiIndex] = sentence
            else:
                for annotation in sentence:
                    pair[wikiIndex][annotation] += sentence[annotation]
    return pair


filename1 = sys.argv[1]
filename2 = sys.argv[2]
matchesFilename = sys.argv[3]
indir = sys.argv[4]
outdir = sys.argv[5]

#read in the lemmatized original matches, split on whitespace, lowercase, remove punctuation
#create lookup set
lookupSet = set()
for filename in (filename1, filename2):
    with open(filename) as f:
        for line in f:
            words = filter(lambda x:x.isalpha(), (x.lower() for x in line.strip().split()))
            lookupSet.add(tuple(sorted(words)))
print(len(lookupSet))

#read in greedy matches file and store lookup for each article
articleLookup = collections.defaultdict(list)
with open(matchesFilename) as f:
    for line in f:
        try:
            articleIndex, sentence1, sentence2, score = line.split('\t')
        except Exception:
            print(line)
            continue
        _index1, _index2 = sentence1[1:-1].split(',')
        if _index2 != '':
            index1 = int(_index1),int(_index2)
        else:
            index1 = (int(_index1),)

        _index1, _index2 = sentence2[1:-1].split(',')
        if _index2 != '':
            index2 = int(_index1),int(_index2)
        else:
            index2 = (int(_index1),)

        articleLookup[int(articleIndex)].append((index1, index2))
        
#read in parsed wikipedia data
#check if any sentences are dupe
#add to parsed pairs
#write to new greedy matches file
parsedPairs = []
for article in parallelWikipedia.iterParsedParallelWikipedia(indir, verbose=True):
    if article['index'] in articleLookup:
        print(article['index'], articleLookup[article['index']])
        for index1,index2 in articleLookup[article['index']]:
            pair = makeParsedPair(index1, index2, article, lookupSet)
            if pair is not None:
                parsedPairs.extend(pair)

print(len(parsedPairs))
#output parsed pairs with _ prefix so they come last
with gzip.open(os.path.join(outdir, '_newParsed.json.gz'), 'w') as f:
    json.dump(parsedPairs, f)

