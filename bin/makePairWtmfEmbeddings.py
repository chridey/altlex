import nltk
import sys

import wtmf

from chnlp.word2vec import sentenceRepresentation

class WTMFEmbeddingsPairs(sentenceRepresentation.WTMFEmbeddings):
    def __init__(self, dataFilename, modelFilename):
        sentenceRepresentation.LocalDataEmbeddings.__init__(self, dataFilename, modelFilename)
        if self._model is None:
            self._model = wtmf.WTMFVectorizer(input='content', tokenizer=lambda x:x)
        self._Q,Q2 = self._model.load(modelFilename)
        
    def iterSentencePairs(self):
        for fileIndex in range(len(self._data['articles'])):
            for articleIndex in range(len(self._data['articles'][fileIndex])):
                sentences = [[j.lower() for j in nltk.word_tokenize(i)] for i in self.lookupSentences(articleIndex, fileIndex)]
                for sent1,sent2 in zip(sentences, sentences[1:]):
                    yield sent1+sent2

if __name__ == '__main__':
    e = WTMFEmbeddingsPairs(sys.argv[2], sys.argv[1])
    Q2 = e._model.transform(e.iterSentencePairs())
    e._model.save(sys.argv[1], e._Q, Q2)
