#go through all the files in every directory

#for those where the number of lines in the file does not match the number of sentences in the XML:
#output to file bad_splits

#for those directories where the number of outfiles does not equal the number of infiles
#output to file unparsed_files

#call read_xml from corenlp

from __future__ import print_function

import sys
import os
import corenlp
import gzip
import json

from altlex.embeddings import sentenceRepresentation

def combineDocuments(docs):
    return corenlp.document.Document(docs)

def makeSentenceData(docs):
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
    return annotatedSentences

inputDir = sys.argv[1]
outputDir = sys.argv[2]
wikiFilename = sys.argv[3]
maxArticles = int(sys.argv[4])
                  
port = 9989
c = corenlp.client.CoreNLPClient(port=port, dep_type='basic-dependencies')
modelFilename = None
sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiFilename,
                                                                modelFilename)

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

#f = open(os.path.join(outputDir, 'unparsed_files'), 'w')
#f2 = open(os.path.join(outputDir, 'good_splits'),'w')

parallelWikis = {'titles': ['' for i in range(maxArticles)],
                 'files': ['en', 'simple'],
                 'articles': [[[] for i in range(maxArticles)],
                              [[] for i in range(maxArticles)]]}
titles = sentRep.titles

dirs = sorted(os.listdir(inputDir))
if len(sys.argv) > 5:
    start = sys.argv[5]
    startIndex = dirs.index(start)
else:
    startIndex = 0
    
for filename in dirs[startIndex:]:
    fullPathDir = os.path.join(inputDir, filename)
    if os.path.isdir(fullPathDir):
        print(filename)
        articleLookup = {}
        total = 0
        minIndex = float('inf')
        maxIndex = -float('inf')

        for filename2 in sorted(os.listdir(fullPathDir)):
            fullPathFilename = os.path.join(fullPathDir, filename2)
            if fullPathFilename[-2:] in ('.0', '.1'):
                if not os.path.exists(fullPathFilename + '.out'):
                    continue #print(fullPathFilename, file=f)
                else:
                    print(filename2)
                    articleIndex = int(filename2[:-2])
                    wikiIndex = int(filename2[-1])

                    if articleIndex < minIndex:
                        minIndex = articleIndex
                    if articleIndex > maxIndex:
                        maxIndex = articleIndex
                    
                    if articleIndex not in articleLookup:
                        articleLookup[articleIndex] = {'index': articleIndex,
                                                       'title': titles[articleIndex],
                                                       'sentences': [None, None]}
                    
                    with open(fullPathFilename) as f3:
                        docs = f3.read().splitlines()

                    with open(fullPathFilename + '.out') as f4:
                        parsed_sentences = corenlp.file_reader.read_xml(f4, dep_type='basic-dependencies')

                    parsed_docs = [combineDocuments([parsed_sentence]) for parsed_sentence in parsed_sentences]
                    sentences = []
                    for parsed_sentence in parsed_sentences:
                        sentences.append(' '.join(token.surface for token in parsed_sentence))
                    parallelWikis['articles'][wikiIndex][articleIndex] = sentences
                    parallelWikis['titles'][articleIndex] = titles[articleIndex]
                    '''
                    total += len(parsed_sentences)

                    split_sentences = []
                    for doc in c.annotate_mp(docs, n_procs=20):
                        tokens = []
                        for sentence in doc.sents:
                            tokens += sentence.tokens
                        split_sentences.append(tokens)

                    parseIndex = 0
                    parsed_docs = []
                    curr_parsed_length = 0
                    curr_split_length = 0
                    total_lost = 0
                    #now go through each tokenized line and make sure the length of the tokens matches the length of the parsed_sentence
                    #if it doesnt, add the current tokens until it is larger than the length and then break
                    print(fullPathFilename, len(docs), len(parsed_sentences), len(split_sentences), file=sys.stderr)
                    for sentenceIndex,sentence in enumerate(split_sentences):
                        
                        if curr_parsed_length != curr_split_length:
                            #print(sentenceIndex, parseIndex, curr_split_length, curr_parsed_length)
                            while curr_parsed_length < curr_split_length and parseIndex < len(parsed_sentences):
                                curr_parsed_length += len(parsed_sentences[parseIndex].tokens)
                                parseIndex += 1
                            if curr_split_length != curr_parsed_length:
                                curr_split_length += len(sentence)
                                parsed_docs.append(combineDocuments([]))
                                total_lost += 1
                                continue

                        if parseIndex >= len(parsed_sentences):
                            parsed_docs.append(combineDocuments([]))
                            total_lost += 1
                            continue
    
                        if len(sentence) == len(parsed_sentences[parseIndex].tokens):
                            parsed_docs.append(combineDocuments([parsed_sentences[parseIndex]]))
                            parseIndex += 1
                        elif len(parsed_sentences[parseIndex].tokens) < len(sentence):
                            #increment parseIndex until it is >=
                            curr_sentences = []
                            curr_parsed_length = 0
                            curr_split_length = 0
                            while curr_parsed_length < len(sentence) and parseIndex < len(parsed_sentences):
                                curr_sentences.append(parsed_sentences[parseIndex])
                                curr_parsed_length += len(parsed_sentences[parseIndex].tokens)
                                parseIndex += 1
                            if curr_parsed_length == len(sentence):
                                parsed_docs.append(combineDocuments(curr_sentences))
                        else:
                            #cant handle this case
                            curr_parsed_length += len(parsed_sentences[parseIndex].tokens)
                            curr_split_length += len(sentence)
                            parseIndex += 1
                            total_lost += 1
                            parsed_docs.append(combineDocuments([]))

                    print('lost: {}'.format(total_lost))
                    '''
                  
                    articleLookup[articleIndex]['sentences'][wikiIndex] = makeSentenceData(parsed_docs)


        if minIndex == float('inf') or maxIndex == -float('inf'):
            continue
        if set(articleLookup)-set(range(minIndex,maxIndex+1)):
            print(fullPathDir, set(articleLookup)-set(range(minIndex,maxIndex+1)), file=sys.stderr)

        j = []
        for articleIndex in range(minIndex,maxIndex+1):
            if articleIndex in articleLookup:
                j.append(articleLookup[articleIndex])
            else:
                j.append({'index': articleIndex, 'sentences': [None, None]})
        outfilename = '{}.{}.{}.json.gz'.format(minIndex, maxIndex+2, total)

        with gzip.GzipFile(os.path.join(outputDir, outfilename), 'w') as f5:
            json.dump(j, f5)

#f.close()
#f2.close()

with gzip.open(os.path.join(outputDir, 'parallelwikis5.json.gz'), 'w') as f:
    json.dump(parallelWikis, f)
