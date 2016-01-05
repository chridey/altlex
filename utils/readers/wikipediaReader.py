#write new sentenceIterator for sentences and pairs
#(see previous and current sentence)

#write new fileReader and sentenceReader (just use gigaword sentence reader)

#TODO: indexed file wikipedia reader

import json
import bz2
import gzip
import nltk
import operator

class ExtractedWikipediaReader:
    def __init__(self, wikipedia_file):
        if wikipedia_file.endswith('.bz2'):
            self.opener = bz2.BZ2File
        else:
            self.opener = open

        self.filename = wikipedia_file

    def iterFiles(self):
        with self.opener(self.filename) as f:
            for line in f:
                #print(line)
                if '</doc>' in line:
                    yield articleName,document
                elif '<doc id=' in line and '>'  in line:
                    document = []
                    articleName = line[line.find('title="')+len('title="'):-3]
                elif not line:
                    continue
                else:
                    if '.' not in line and '!' not in line and '?' not in line:
                        continue

                    document.extend(nltk.sent_tokenize(line.decode('utf-8')))

class ParallelDataReader:
    def __init__(self, *filenames):
        for filename in filenames:
            with gzip.GzipFile(filename) as f:
                self._data = json.load(f)

    def setData(self, whichFile):
        self._whichFile = whichFile

    def iterFiles(self):
        for index,title in enumerate(self._data['titles']):
            yield title,self._data['articles'][self._whichFile][index]

class ParallelWikipediaReader_T:
    def __init__(self, embeddings):
        self._embeddings = embeddings
        print(len(self._data['en']), len(self._data['simple']))
        print(model.numElements)
        print(self._data['enstarts'][-1], self._data['simplestarts'][-1])
        assert(model.numElements == self._data['enstarts'][-1] + self._data['simplestarts'][-1])

    def getSentenceRepresentations(self, title, pairs=False):
        assert(self._model is not None)
        titleIndex = self._titleLookup[title]

        enOrigIndex = self._data['enorig'][titleIndex]
        enStart = self._data['enstarts'][enOrigIndex]
        enEnd = self._data['enstarts'][enOrigIndex+1]
        vecs1 = self._model.sentenceRepresentation(*[i for i in range(enStart,enEnd)])
        #print(enOrigIndex, enStart, enEnd, len(vecs1), len(self._data['en'][self._data['enorig'][titleIndex]]))
        if pairs:
            en = map(nltk.word_tokenize, self._data['en'][self._data['enorig'][titleIndex]])
            vecs1 += self._model.inferSentences(*list(map(operator.add, en[:-1], en[1:])))
            
        totalEnSentences = self._data['enstarts'][-1]
        simpleOrigIndex = self._data['simpleorig'][titleIndex]
        simpleStart = self._data['simplestarts'][simpleOrigIndex] + totalEnSentences
        simpleEnd = self._data['simplestarts'][simpleOrigIndex+1] + totalEnSentences
        vecs2 = self._model.sentenceRepresentation(*[i for i in range(simpleStart,simpleEnd)])
        if pairs:
            simple = map(nltk.word_tokenize, self._data['simple'][self._data['simpleorig'][titleIndex]])
            vecs2 += self._model.inferSentences(*list(map(operator.add, simple[:-1], simple[1:])))

        print(len(vecs1), len(vecs2))
        return vecs1, vecs2

