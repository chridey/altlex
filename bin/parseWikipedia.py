import os
import sys
import gzip
import json
import time

import corenlp

from chnlp.word2vec import sentenceRepresentation

if __name__ == '__main__':
    port = 9989
    numProcs = 8
    rev = False
     
    c = corenlp.client.CoreNLPClient(port=port)
    
    wikiFilename = sys.argv[1]
    outputDir = sys.argv[2]
    if len(sys.argv) > 3:
        startIndex = int(sys.argv[3])
    else:
        startIndex = 0
    
    modelFilename = None
    sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiFilename,
                                                                    modelFilename)

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
        
    total = 0
    prevIndex = startIndex
    j = []
    starttime = time.time()
    timeBetweenOutput = 60*60*2

    titles = list(enumerate(sentRep.titles))
    if len(sys.argv) > 4:
        endIndex = int(sys.argv[4])
    else:
        endIndex = len(titles)-1
    if rev:
        titles = titles[::-1]
        prevIndex = endIndex
        
    for index,title in titles:
        if index < startIndex or index > endIndex:
            continue
        print('Index: {} Title: {} Total: {}'.format(index, title.encode('utf-8'), total))
        pair = {'title': title, 'sentences': []}
        for fileIndex in range(2):
            sentences = sentRep.getSentences(index, fileIndex)
            print(len(sentences))
            total += len(sentences)
            docs = c.annotate_mp(sentences, n_procs=numProcs)
            if docs is None:
                print('Something went wrong!')
                break
            annotatedSentences = []
            for doc in docs:
                annotatedSentence = {'parse': [],
                                     'words': [],
                                     'lemmas': [],
                                     'pos': [],
                                     'ner': [],
                                     'dep': []}
                for sent in doc.sents:
                    annotatedSentence['parse'].append(str(sent.parse))
                    depTriples = [(gov.index, dep.index, rel)
                                  for gov in sent.gov2deps
                                  for rel,dep in sent.gov2deps[gov]]
                    annotatedSentence['dep'].append(depTriples)
                    words = []
                    lemmas = []
                    pos = []
                    ner = []
                    for token in sent.tokens:
                        words.append(token.surface)
                        lemmas.append(token.lem)
                        pos.append(token.pos)
                        ner.append(token.ne)
                    annotatedSentence['words'].append(words)
                    annotatedSentence['lemmas'].append(lemmas)
                    annotatedSentence['pos'].append(pos)
                    annotatedSentence['ner'].append(ner)
                annotatedSentences.append(annotatedSentence)
            pair['sentences'].append(annotatedSentences)

        if rev:
            j.insert(0, pair)
        else:
            j.append(pair)
                
        #need to output results every so often
        if time.time() - starttime > timeBetweenOutput or index == len(titles)-1 or index==endIndex:
            if rev:
                filename = '{}.{}.{}.json.gz'.format(index, prevIndex, total)
                prevIndex = index + 1
            else:
                filename = '{}.{}.{}.json.gz'.format(prevIndex, index, total)
                prevIndex = index - 1

            with gzip.GzipFile(os.path.join(outputDir, filename), 'w') as f:
                json.dump(j, f)
            j = []
            starttime = time.time()
