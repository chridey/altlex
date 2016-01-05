import time

import json
import bz2
import nltk

#TODO: create wikipedia sentence/paired sentence iterator
#first iterate over sentences, then over adjacent sentence pairs
#ID is the document name and the usual sentence ID (see wikipedia.py)

class LabeledLineSentence(object):
    def __init__(self,
                 filenames,
                 reader,
                 sentenceReader,
                 sentenceType,
                 maxTime=float('inf'),
                 maxPoints=float('inf'),
                 matcher=None,
                 filterFiles=None,
                 pairs=False,
                 labels=False,
                 verbosity=1):
        self.filenames = filenames
        self.reader = reader
        self.sentenceReader = sentenceReader
        self.sentenceType = sentenceType
        self.maxTime = maxTime
        self.maxPoints = maxPoints
        self.matcher = matcher
        self.filterFiles = filterFiles
        self.pairs = pairs
        self.labels = labels
        self.verbosity = verbosity
        
        self.articles = [{} for i in xrange(len(filenames))]
        self.usedArticles = [[] for i in xrange(len(filenames))]

    def iterFileReaders(self):
        for filename in self.filenames:
            if self.verbosity>=1:
                print(filename)
            yield self.reader(filename)
        
    def __iter__(self):
        for fileIndex,reader in enumerate(self.iterFileReaders()):

            startTime = time.time()
            numPoints = 0
            
            for nameIndex,(name,sentences) in enumerate(reader.iterFiles()):
                if self.verbosity>=1:
                    print(name.encode('utf-8'))
                if self.verbosity>=2:
                    print(numPoints)

                if fileIndex >= len(self.articles) :
                    self.articles.append({})
                self.articles[fileIndex][name] = nameIndex
                
                if self.filterFiles is not None and name not in self.filterFiles:
                    continue

                if fileIndex >= len(self.usedArticles) :
                    self.usedArticles.append([])
                self.usedArticles[fileIndex].append(name)
                
                sr = self.sentenceReader(sentences)
                prevSentence = None


                #this ignores all one sentence articles
                for index,sentence in enumerate(sr.iterSentences()): #parse=False):
                    if self.verbosity>=3:
                        print(sentence.words)

                    if self.matcher is None or self.matcher.matches(sentence.words):
                        numPoints += 1
                        if self.labels:
                            yield self.sentenceType(sentence.nelemmas,
                                                    tags=['SENT_{}_{}_{}_0'.format(fileIndex,
                                                                                   len(self.usedArticles[fileIndex]),
                                                                                   index)])
                        else:
                            yield self.sentenceType(sentence.nelemmas)
                        if self.pairs and prevSentence is not None:
                            if self.labels:
                                yield self.sentenceType(prevSentence.nelemmas + sentence.nelemmas,
                                                        tags=['SENT_{}_{}_{}_1'.format(fileIndex,
                                                                                       len(self.usedArticles[fileIndex]),
                                                                                       index-1)])
                            else:
                                yield self.sentenceType(prevSentence.nelemmas + sentence.nelemmas)

                        #yield prevSentence, sentence, fileIndex, len(self.usedArticles[fileIndex]), index

                    prevSentence = sentence

                if numPoints > self.maxPoints or time.time()-startTime > self.maxTime:
                    break

            print(numPoints)

    def save(self, filename):
        with bz2.BZ2File(filename, 'w') as f:
            json.dump({'files': self.filenames,
                       'articles': self.articles,
                       'usedArticles': self.usedArticles},
                      f)

    def load(self, filename):
        with bz2.BZ2File(filename) as f:
            j = json.load(f)
        self.filenames = j['files']
        self.articles = j['articles']
        self.usedArticles = j['usedArticles']
        
    def getArticle(self, fileIndex, articleIndex):
        r = self.reader(self.filenames[fileIndex])
        currentIndex = 0
        searchFile = self.articles[fileIndex][articleIndex]
        for nameIndex,(name,sentences) in enumerate(r.iterFiles()):
            #print(name, nameIndex, currentIndex, articleIndex)
            if name == searchFile:
                return sentences

    def getArticleIndex(self, fileIndex, articleName):
        r = self.reader(self.filenames[fileIndex])
        currentIndex = 0

        for nameIndex,(name,sentences) in enumerate(r.iterFiles()):
            if name == articleName:
                return nameIndex

    def getSentence(self, fileIndex, articleIndex, sentenceIndex):
        s = self.getArticle(fileIndex, articleIndex)
        return s[sentenceIndex]
            
class Sentence(list):
    def __init__(self, words, *args, **kwargs):
        self[:] = words

class PreviousSentence(LabeledLineSentence):
    def __iter__(self):
        for prevSentence,sentence,fid,aid,sid in LabeledLineSentence.__iter__(self):
            if prevSentence is None:
                continue
            yield self.sentenceType(prevSentence.nelemmas,
                                    labels=['SENT_{}_{}_{}'.format(fid, aid, sid)])

class CurrentSentence(LabeledLineSentence):
    def __iter__(self):
        for prevSentence,sentence,fid,aid,sid in LabeledLineSentence.__iter__(self):
            #print(sentence.nelemmas)
            yield self.sentenceType(sentence.nelemmas,
                                    labels=['SENT_{}_{}_{}'.format(fid, aid, sid)])

#z=? bits for file name. 14?
#a=23 bits for article name. 28?
#b=? bits for sentence id. 20?
#c=1 bit for pair or single sentence
def packBits(fid, aid, sid, pid):
    fBits = 14
    aBits = 28
    sBits = 20
    cBits = 1
    return fid << aBits+sBits+cBits | aid << sBits+cBits | sid << cBits | pid

class ParallelSentences(LabeledLineSentence):
    def iterFileReaders(self):
        reader = self.reader(*self.filenames)
        for i in range(2):
            reader.setData(i)
            yield reader
    
class SentencePair(LabeledLineSentence):
    def __iter__(self):
        for prevSentence,sentence,fid,aid,sid in LabeledLineSentence.__iter__(self):
            yield self.sentenceType(sentence.nelemmas,
                                    tags=['SENT_{}_{}_{}_0'.format(fid, aid, sid)])

            if prevSentence is not None:
                yield self.sentenceType(prevSentence.nelemmas + sentence.nelemmas,
                                        tags=['SENT_{}_{}_{}_1'.format(fid, aid, sid-1)])

class UnlabeledSentencePair(LabeledLineSentence):
    def __iter__(self):
        for prevSentence,sentence,fid,aid,sid in LabeledLineSentence.__iter__(self):
            yield self.sentenceType(sentence.nelemmas)

            if prevSentence is not None:
                yield self.sentenceType(prevSentence.nelemmas + sentence.nelemmas)

class UnlabeledSentence(LabeledLineSentence):
    def __iter__(self):
        for prevSentence,sentence,fid,aid,sid in LabeledLineSentence.__iter__(self):
            yield self.sentenceType(sentence.nelemmas)                                        


        
        
