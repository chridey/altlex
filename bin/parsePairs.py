import os
import sys
import gzip
import json
import time

import corenlp

if __name__ == '__main__':
    port = 9989
    numProcs = 8
     
    c = corenlp.client.CoreNLPClient(port=port)
    
    pairsFilename = sys.argv[1]
    outputDir = sys.argv[2]
    chunkSize = int(sys.argv[3])
    if len(sys.argv) > 4:
        startIndex = int(sys.argv[4])
    else:
        startIndex = 0

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
        
    total = 0
    prevIndex = startIndex
    j = []
    starttime = time.time()

    lines = []
    with open(pairsFilename) as f:
        for line in f:
            try:
                sent1, sent2, score = line.strip().decode('utf-8').split('\t')
            except ValueError:
                continue
            lines.append([sent1, sent2, score])

    chunk = []
    prevIndex = startIndex
    for index,line in enumerate(lines[startIndex:]):
        chunk.extend([line[0], line[1]])
        if len(chunk) >= chunkSize:
            print(index+startIndex)
            docs = c.annotate_mp(chunk, n_procs=numProcs)
            if docs is None:
                print('Something went wrong!')
                break
            assert(len(docs) == len(chunk))
            pair = {'title': index+startIndex, 'sentences': []}
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

            j.append(pair)
                
            #need to output results every so often
            filename = '{}.{}.json.gz'.format(prevIndex, index+startIndex)
            prevIndex = index+startIndex - 1

            with gzip.GzipFile(os.path.join(outputDir, filename), 'w') as f:
                json.dump(j, f)
            j = []
            chunk = []
    print(time.time() - starttime)
