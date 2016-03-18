import gzip
import json
import base64
import logging

#non-standard packages
import numpy as np
import gensim
import requests
from sklearn.metrics import pairwise_distances

#local packages
#import wtmf

logger = logging.getLogger(__file__)

class PairedSentenceEmbeddings:
    cache = {}
    
    def __init__(self, dataFilename, modelFilename):
        if dataFilename in PairedSentenceEmbeddings.cache:
            logger.debug('Loading data from cache')
            self._data = PairedSentenceEmbeddings.cache[dataFilename]
        else:
            logger.debug('Reading data from file')            
            self.loadData(dataFilename)
            PairedSentenceEmbeddings.cache[dataFilename] = self._data
            
        if modelFilename in PairedSentenceEmbeddings.cache:
            logger.debug('Loading model from cache')
            self._model = PairedSentenceEmbeddings.cache[modelFilename]
        else:
            self._model = None

    def loadData(self, dataFilename):
        with gzip.GzipFile(dataFilename) as f:
            self._data = json.load(f)
        self._titleLookup = {j:i for (i,j) in enumerate(self._data['titles'])}
        logger.debug(self._data['files'])
        logger.debug("Total Articles: {}".format(len(self._data['titles'])))
        #logger.debug("Max Indices: {}, {}".format(self._data['starts'][0][-1],
        #                                          self._data['starts'][1][-1]))
        logger.debug("Total Sentences: {}, {}".format(sum(map(len, self._data['articles'][0])),
                                                      sum(map(len, self._data['articles'][1]))))
        
    @property
    def titles(self):
        return self._data['titles']
    
    def iterTitles(self):
        for title in self.titles:
            yield title 

    def lookupSentences(self, articleIndex, fileIndex):
        return self._data['articles'][fileIndex][articleIndex]

    def getSentences(self, articleIndex, fileIndex, pairs=False):
        sentences = self.lookupSentences(articleIndex, fileIndex)
        #print(len(sentences))
        sentencePairs = []
        if pairs:
            sentencePairs += list(map(' '.join, zip(sentences, sentences[1:])))
        #print(len(sentencePairs))
        return sentences + sentencePairs

    def getSentencePairs(self, articleIndex, pairs=False):
        return (self.getSentences(articleIndex, 0, pairs),
                self.getSentences(articleIndex, 1, pairs))

    def lookupEmbeddings(self, articleIndex, fileIndex):
        raise NotImplementedError

    def lookupPairEmbeddings(self, articleIndex, fileIndex):
        raise NotImplementedError

    def getSentenceEmbeddings(self, articleIndex, fileIndex, pairs=False):
        embeddings = self.lookupEmbeddings(articleIndex, fileIndex)
        embeddingPairs = []
        if pairs:
            embeddingPairs += self.lookupPairEmbeddings(articleIndex, fileIndex)
        return embeddings + embeddingPairs
    
    def getSentenceEmbeddingPairs(self, articleIndex, pairs=False):
        return (self.getSentenceEmbeddings(articleIndex, 0, pairs),
                self.getSentenceEmbeddings(articleIndex, 1, pairs))
    
    #method for looking up a sentence embedding given a file, article, and sentence index
    def getEmbedding(self, fileIndex, articleIndex, sentenceIndex):
        raise NotImplementedError

    def infer(self, words):
        raise NotImplementedError

    def batchSimilarity(self, index, pairs=False, verbose=False, n_jobs=8, metric='cosine'):
        ret = []

        if verbose:
            print(index)
        vecs = self.getSentenceEmbeddingPairs(index, pairs=pairs)
        X = np.asarray(vecs[0])
        Y = np.asarray(vecs[1])
        m = 1-pairwise_distances(X, Y, metric, n_jobs)
        if verbose:
            print(m.shape)
        ret.extend(m.ravel().tolist())

        return ret
    
class LocalDataEmbeddings(PairedSentenceEmbeddings):
    def __init__(self, dataFilename, modelFilename):
        PairedSentenceEmbeddings.__init__(self, dataFilename, modelFilename)

    def lookupEmbeddings(self, articleIndex, fileIndex):
        return [self.getEmbedding(fileIndex, articleIndex, i)
                for i in range(len(self._data['articles'][fileIndex][articleIndex]))]

    def lookupPairEmbeddings(self, articleIndex, fileIndex):
        return [self.getEmbedding(fileIndex, articleIndex, (i,i+1))
                for i in range(len(self._data['articles'][fileIndex][articleIndex])-1)]

class WTMFEmbeddings(LocalDataEmbeddings):
    '''Class for storing the Q matrix (latent document representations) as well as the trained model'''
    def __init__(self, dataFilename, modelFilename):
        LocalDataEmbeddings.__init__(self, dataFilename, modelFilename)
        if self._model is None:
            self._model = wtmf.WTMFVectorizer(input='content', tokenizer=lambda x:x)
        Q,Q2 = self._model.load(modelFilename)
        print(Q.shape, Q2.shape)
        self._sentenceEmbeddings = []
        self._pairwiseEmbeddings = []
        sentPtr = 0
        pairPtr = 0
        for fileIndex in range(len(self._data['articles'])):
            self._sentenceEmbeddings.append([])
            self._pairwiseEmbeddings.append([])
            for articleIndex in range(len(self._data['articles'][fileIndex])):
                length = len(self.lookupSentences(articleIndex, fileIndex))
                self._sentenceEmbeddings[fileIndex].append(Q[sentPtr:sentPtr+length])
                self._pairwiseEmbeddings[fileIndex].append(Q2[pairPtr:pairPtr+length-1])
                sentPtr += length
                pairPtr += (length-1)
        print(sentPtr, pairPtr)
        
    def lookupEmbeddings(self, articleIndex, fileIndex):
        return self._sentenceEmbeddings[fileIndex][articleIndex]

    def lookupPairEmbeddings(self, articleIndex, fileIndex):
        return self._pairwiseEmbeddings[fileIndex][articleIndex]
    
    #method for looking up a sentence embedding given a file, article, and sentence index
    def getEmbedding(self, fileIndex, articleIndex, sentenceIndex):
        if type(sentenceIndex) == int:
            return self._sentenceEmbeddings[fileIndex][articleIndex][sentenceIndex]
        return self._pairwiseEmbeddings[fileIndex][articleIndex][sentenceIndex[0]]
    
        '''
        if type(sentenceIndex) == int:
            #plus start of articles
            if fileIndex == 0:
                start = 0
            elif fileIndex == 1:
                start = self._data['starts'][0][-1]
            return self._Q[start+self._data['starts'][fileIndex][articleIndex]+sentenceIndex]
        data = ' '.join((self._data['articles'][fileIndex][articleIndex][sentenceIndex[0]],
                        self._data['articles'][fileIndex][articleIndex][sentenceIndex[1]]))
        return self._model.transform([nltk.word_tokenize(data)])[0]
        '''
        
class Doc2VecEmbeddings(LocalDataEmbeddings):
    def __init__(self, dataFilename, modelFilename):
        LocalDataEmbeddings.__init__(self, dataFilename, modelFilename)
        if self._model is None:
            self._model = gensim.models.doc2vec.Doc2Vec.load(modelFilename)

    def getEmbedding(self, fileIndex, articleIndex, sentenceIndex):
        if type(sentenceIndex) != int:
            assert(sentenceIndex[1] == sentenceIndex[0]+1)            
            sentenceIndex = sentenceIndex[0]
            isPair = 1
        else:
            isPair = 0
        return self._model.docvecs['SENT_{}_{}_{}_{}'.format(fileIndex,
                                                             articleIndex+1,
                                                             sentenceIndex,
                                                             isPair)]

    def infer(self, words):
        return self._model.infer_vector(words)

class Word2VecEmbeddings(LocalDataEmbeddings):
    def __init__(self, dataFilename, modelFilename):
        LocalDataEmbeddings.__init__(self, dataFilename, modelFilename)
        if self._model is None:
            self._model = gensim.models.word2vec.Word2Vec.load(modelFilename)
        self._oov = {}
        
    def loadData(self, dataFilename):
        self._data = None
    
    def lookupWordEmbedding(self, word):
        if word in self._model:
            return self._model[word]
        elif word not in self._oov:
            try:
                logger.debug('word {} not in model'.format(word.encode('utf-8')))
            except Exception:
                pass
            self._oov[word] = np.random.uniform(-.2, .2, self._model.vector_size)
        return self._oov[word]
        
def factory(data, model):
    if not model or model == 'None':
        return PairedSentenceEmbeddings(data, None)
    elif 'wtmf' in model:
        return WTMFEmbeddings(data, model)
    elif 'doc2vec' in model:
        return Doc2VecEmbeddings(data, model)
    elif 'word2vec' in model:
        return Word2VecEmbeddings(data, model)
    else:
        raise NotImplementedError
    


class PairedSentenceEmbeddingsClient(PairedSentenceEmbeddings):
    def __init__(self, dataFilename, modelFilename, host='localhost', port=5000):
        self._dataFilename = dataFilename
        self._modelFilename = modelFilename
        self._host = host
        self._port = port

    def makeURL(self, joinString, **kwargs):
        ret = 'http://{}:{}/{}?{}={}&{}={}'.format(self._host,
                                            self._port,
                                            joinString,
                                            'data',
                                            self._dataFilename,
                                            'model',
                                            self._modelFilename)
        for k,v in kwargs.iteritems():
            ret += '&{}={}'.format(k,v)
        return ret
    
    @property
    def titles(self):
        r = requests.get(self.makeURL('titles'))
        return r.json()
    
    def lookupSentences(self, articleIndex, fileIndex):
        r = requests.get(self.makeURL('sentences',
                                      articleIndex=articleIndex,
                                      fileIndex=fileIndex))
        return r.json()

    def lookupEmbeddings(self, articleIndex, fileIndex):
        r = requests.get(self.makeURL('embeddings',
                                      articleIndex=articleIndex,
                                      fileIndex=fileIndex))
        return r.json()
    
    def lookupPairEmbeddings(self, articleIndex, fileIndex):
        r = requests.get(self.makeURL('pairEmbeddings',
                                      articleIndex=articleIndex,
                                      fileIndex=fileIndex))
        return r.json()

    def lookupWordEmbedding(self, word):
        try:
            encodedWord = base64.b64encode(word.encode('utf-8'))
        except UnicodeDecodeError:
            return []
        r = requests.get(self.makeURL('wordEmbedding',
                                      word=encodedWord))
        try:
            return r.json()
        except ValueError:
            #try again once more
            encodedWords = base64.b64encode(word.encode('utf-8'))
            r = requests.get(self.makeURL('wordEmbedding',
                                          word=encodedWord))

            try:
                return r.json()
            except ValueError:
                return []
    
    def infer(self, words):
        try:
            encodedWords = base64.b64encode(' '.join(words))
        except UnicodeEncodeError:
            return []
        r = requests.get(self.makeURL('infer',
                                      words=encodedWords))
        try:
            return r.json()
        except ValueError:
            #try again once more
            encodedWords = base64.b64encode(' '.join(words))
            r = requests.get(self.makeURL('infer',
                                          words=encodedWords))
            try:
                return r.json()
            except ValueError:
                return []

    def batchSimilarity(self,
                        articleIndex,
                        pairs=False,
                        verbose=False,
                        n_jobs=8,
                        metric='cosine',
                        side='server'):
        if side == 'server':
            r = requests.get(self.makeURL('batchSimilarity',
                                          articleIndex=articleIndex,
                                          njobs=n_jobs,
                                          pairs=pairs,
                                          ))
            return r.json()
        elif side == 'client':
            #run on client side
            return super(PairedSentenceEmbeddingsClient,
                         self).batchSimilarity(index,
                                               pairs=pairs,
                                               verbose=verbose,
                                               n_jobs=n_jobs,
                                               metric=metric)
