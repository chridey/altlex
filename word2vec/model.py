import sys

from nltk.stem import SnowballStemmer

if sys.version_info > (3,):
    from functools import lru_cache
    import json
else:
    from functools32 import lru_cache
    import gensim

class Model:
    sn = SnowballStemmer('english')

    def __init__(self, filename):
        self.filename = filename
        #load model lazily
        self._model = None
        if 'words' in self.filename:
            self.wordType = 'word'
        elif 'lemmas' in self.filename:
            self.wordType = 'lemma'
        elif 'stems' in self.filename:
            self.wordType = 'stem'
        else:
            raise NotImplementedError
        print(self.wordType)

    def _load(self):
        if self._model is None:
            if sys.version_info < (3,):
                self._model = gensim.models.Word2Vec.load(self.filename)
            else:
                with open(self.filename + '.json') as f:
                    self._model = json.load(f)
                pass #TODO

    @lru_cache(maxsize=None)
    def getNeighbors(self, word, lemma, stem):
        self._load()
        if self.wordType == 'word':
            entity = word
        elif self.wordType == 'lemma':
            entity = lemma
        else:
            entity = stem
        try:
            return [(self.sn.stem(e[0]),e[1]) for e in self._model.most_similar(entity)]
        except KeyError:
            return [(self.sn.stem(entity), 1)]

    def vector(self, word):
        self._load()
        try:
            return self._model[word]
        except KeyError:
            return None
