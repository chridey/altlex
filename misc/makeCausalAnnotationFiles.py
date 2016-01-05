from __future__ import print_function

#first read from paraphrases
#create sentence lookup
#then read from extracted files
#sort categories by harmonic mean of avg0 and avg1
#then iterate through sentence embeddings
#have at least n=20 sentences each?
#and at least X=1 causal markers?
#and at least y=5 paraphrases?

import sys
import collections
import os
import gzip
import json
import time

from chnlp.misc import wiknet
from chnlp.word2vec import sentenceRepresentation

minSentences = 20
minCausal = 1
minParaphrase = 5

if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)

    paraphraseFile = sys.argv[1]
    extractedFile = sys.argv[2]
    extractedFileStats = sys.argv[3]
    parsedWikiDir = sys.argv[4]
    wikiLookup = sys.argv[5]
    outDir = sys.argv[6]

    sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiLookup,
                                                                    None)
    titleIndices = {j:i for i,j in enumerate(sentRep.titles)}
    
    paraphrases = set()
    print('reading paraphrases...')
    with open(paraphraseFile) as f:
        for line in f:
            try:
                s1, s2, score = line.split("\t")
            except ValueError:
                continue
            paraphrases.add(s1)
            paraphrases.add(s2)
    print('read {} unique paraphrases'.format(len(paraphrases)))
    
    catFound = False
    categories = {}
    print('reading stats...')
    with open(extractedFileStats) as f:
        for line in f:
            if catFound:
                cat, num, avg0, avg1, avgBoth = line.split("\t")
                avg0, avg1 = float(avg0), float(avg1)
                try:
                    harmonicMean = (2*avg0*avg1)/(avg0+avg1)
                except ZeroDivisionError:
                    harmonicMean = 0
                categories[cat] = harmonicMean
            elif line.startswith('CATEGORIES:'):
                catFound = True
    print('read {} categories'.format(len(categories)))
    
    articles = collections.defaultdict(list)
    titles = {}
    print('reading article stats...')
    with open(extractedFile) as f:
        for line in f:
            title, cat, filename, num0, num1, causal0, causal1 = line.split("\t")
            articles[cat].append(title)
            titles[title] = None
    print('read {} article categories and {} articles'.format(len(articles), len(titles)))

    print('reading parsed wikipedia...')
    if not os.path.exists('overlapping_articles.json.gz'):
        for filename in os.listdir(parsedWikiDir):
            if filename.endswith('.gz'):
                with gzip.open(os.path.join(parsedWikiDir, filename)) as f:
                    j = json.load(f)
                for p in j:
                    if p['title'] in titles:
                        
                        if len(p['sentences'][0]) < minSentences or len(p['sentences'][1]) < minSentences:
                            continue
                        if sum(titles[i] is not None for i in titles) % 500 == 0:
                            print(p['title'], sum(titles[i] is not None for i in titles))
                        titles[p['title']] = p['sentences']

        with gzip.open('overlapping_articles.json.gz', 'w') as f:
            json.dump(titles, f)
    else:
        print(time.time())
        with gzip.open('overlapping_articles.json.gz') as f:
            titles = json.load(f)
        print(time.time())
    print('found {} files'.format(sum(titles[i] is not None for i in titles)))
        
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    count = 0
    for cat in sorted(categories, reverse=True, key=lambda x:categories[x]):
        if categories[cat] <= 0:
            break
        print(cat, categories[cat])
        for title in articles[cat]:
            if not titles[title]:
                continue
            titleIndex = titleIndices[title]
            print(title, titleIndex)
            if count % 100 == 0:
                outfile = os.path.join(outDir,
                                       str(count))
                if not os.path.exists(outfile):
                    os.mkdir(outfile)
            count += 1
            
            if len(titles[title][0]) < minSentences or len(titles[title][1]) < minSentences:
                continue
            sentencesList = []
            numParaphrases = [0, 0]
            numCausal = [0, 0]
            for wikiIndex,wiki in enumerate(titles[title]):
                sentences = []
                for sentence in wiki:
                    words = wiknet.getLemmas(sentence['words'])
                    sentence = ' '.join(words)
                    multisentence = ' '.join((sentence, sentences[-1])) if len(sentences) else sentence
                    if sentence in paraphrases or multisentence in paraphrases:
                        numParaphrases[wikiIndex] += 1
                    numCausal[wikiIndex] += 'because' in sentence
                    sentences.append(sentence)
                sentencesList.append(sentences)

            if numParaphrases[0] < minParaphrase or numParaphrases[1] < minParaphrase or numCausal[0] < minCausal or numCausal[1] < minCausal:
                continue
            for wikiIndex,sentences in enumerate(sentencesList):
                with open(os.path.join(outfile,
                                       '{}.{}'.format(titleIndex,
                                                      wikiIndex)),
                          'w') as f:
                    print('Title: {}'.format(title.encode('utf-8')), file=f)
                    try:
                        print(cat.encode('utf-8'), file=f)
                    except Exception:
                        print(file=f)
                    for sentenceIndex,sentence in enumerate(sentences):
                        words = sentence.split()
                        if len(words) < 4 or ('Template' in sentence and 'has been incorrectly substituted' in sentence):
                            print(file=f)
                            continue

                        print('{}: {}'.format(sentenceIndex, sentence.encode('utf-8')), file=f)
