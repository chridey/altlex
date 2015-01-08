import sqlite3
import itertools
import re

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DiscourseInstance:
    wnl = WordNetLemmatizer()
    sn = SnowballStemmer("english")
    stop = stopwords.words("english")
    
    def __init__(self, relation, klass, firstarg, connective, cattr, secondarg, stem=True, stop=True, lemmatize=True):
        self.relation = relation.decode('latin')
        self.klass = klass.decode('latin')
        changes = {'firstarg': firstarg,
                   'connective': connective,
                   'cattr': cattr,
                   'secondarg': secondarg}
        for clause in changes:
            result = word_tokenize(changes[clause].decode('latin').lower())
            if stop:
                result = [i for i in result if i not in self.stop]
            if lemmatize:
                result = [self.wnl.lemmatize(i) for i in result]
            if stem:
                result = [self.sn.stem(i) for i in result]
                
            #result = [self.sn.stem(self.wnl.lemmatize(i)) for i in result if i not in self.stop]
            #result = R.sub('', changes[clause].lower())
            setattr(self, clause, result)

    @property
    def all_words(self):
        return self.firstarg  + self.connective + self.cattr + self.secondarg

    @property
    def all_bigrams(self):
        words = self.all_words
        return [(words[i],words[i+1]) for i in range(len(words)-1)]

    @property
    def second_arg_bigrams(self):
        return [(words[i],words[i+1]) for i in range(len(self.secondarg)-1)]
        
    def __repr__(self):
        return ' '.join(self.all_words)

class Corpus:
    def __init__(self):
        self.di = []
        
    def add_instance(self, di):
        self.di.append(di)

    @property
    def words(self):
        if not getattr(self, '_words', None):
            self._words = list(itertools.chain(*(i.all_words for i in self.di)))
        return self._words

    @property
    def unique_words(self):
        if not getattr(self, '_unique_words', None):
            self._unique_words = set(self._words)
        return self._unique_words

class CorpusBuilder:
    def __init__(self, db='pdtb.sqlite3'):
        self.conn = sqlite3.connect(db)
        self.c = self.conn.cursor()
        self.conn.text_factory = bytes        
        
    def extract(self, klass=None, relation=None, stem=True, lemmatize=True, stop=True):
        c = Corpus()

        query = "select relation, firstsemfirst, arg1rawtext || ' ' || arg1rawtexta || ' ' || sup1rawtext, rawtext, relrawtext, arg2rawtext || ' ' || arg2rawtexta || ' ' || sup2rawtext from annotations"

        if klass is None:
            if relation is None:
                self.c.execute(query)
            else:
                self.c.execute(query + " where relation = ?", (relation,))
        else:
            if relation is None:
                self.c.execute(query + " where firstsemfirst like ?", (klass,))
            else:
                self.c.execute(query + " where relation = ? and firstsemfirst like ?",(relation, klass))

        for i in self.c:
            c.add_instance(DiscourseInstance(*i, stem=stem, lemmatize=lemmatize, stop=stop))

        return c
