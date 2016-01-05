import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer

DEBUG = 0

class Sentence:
    sn = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    patterns =  '''NP: {<DT|PP\$>?<JJ>*<NN>*}
                  {<NNP>+}
                  {<NN>+}'''
    NPChunker = nltk.RegexpParser(patterns)
    
    def __init__(self,
                 words,
                 lemmas=None,
                 pos=None,
                 ner=None,
                 parse=None,
                 dependencies=None,
                 force=False,
                 chunk=False):
        
        self.words = words

        try:
            self.stems = [self.sn.stem(i) for i in words]
        except UnicodeDecodeError:
            self.words = []
            self.lemmas = []
            self.pos = []
            self.ner = []
            self.parse = None
            self._nelemmas = None
            return

        if pos is None or force:
            self.pos = list(zip(*nltk.pos_tag(self.words)))[1]
        else:
            self.pos = pos

        self.lemmas = []
        try:
            for i,l in enumerate(words):
                if self.pos[i].lower().startswith('v'):
                    o = l.lower()
                    if o == "'s":
                        self.lemmas.append('is')
                    else:
                        self.lemmas.append(self.wnl.lemmatize(o, 'v'))
                elif self.pos[i].lower().startswith('n'):
                    self.lemmas.append(self.wnl.lemmatize(l.lower()))
                #elif self.pos[i].lower().startswith('j'):
                #    self.lemmas.append(self.wnl.lemmatize(l.lower(), 'a'))
                #elif self.pos[i].lower().startswith('r'):
                #    self.lemmas.append(self.wnl.lemmatize(l.lower(), 'r'))
                else:
                    self.lemmas.append(l.lower())
                    
        except UnicodeDecodeError:
            self.words = []
            self.lemmas = []
            self.pos = []
            self.ner = []
            self.parse = None
            self._nelemmas = None
            return
        #self.lemmas = [self.wnl.lemmatize(i) for i in lemmas]
            
        self.ner = ner
        self.parseString = parse
        if parse is not None:
            self.parse = nltk.Tree.fromstring(parse)
        elif chunk:
            self.parse = self.NPChunker.parse(list(zip(self.words, self.pos)))
        else:
            self.parse = None

        self.dependencies = dependencies

        self._nelemmas = None

    def split(self):
        return self.words

    #get either the next lemma or the full named entity
    def nextLemmaAndPOS(self, index, maxIndex):
        lemma = ''
        pos = ''
        while index < maxIndex and \
                  self.ner[index] != 'O':
            if len(lemma):
                lemma += ' '
            lemma += self.words[index].lower()
            pos = self.ner[index]
            index += 1
        if not len(lemma):
            lemma = self.lemmas[index].lower()
            pos = self.pos[index]
            index += 1
        return index,lemma,pos

    @property
    def nelemmas(self):
        if self._nelemmas is None:
            self._nelemmas = []
            i = 0
            while i < len(self.words):
                i,lemma1,pos1 = self.nextLemmaAndPOS(i,
                                                     len(self.words))
                self._nelemmas.append(lemma1)

        return self._nelemmas
    
class SentenceReader:
    def __init__(self, xmlroot):
        self.root = xmlroot
        self.sentenceType = Sentence
        self.document = 'document'
        
    def sentenceExtractor(self, sentence, parse=True):
        assert(sentence.tag == 'sentence')
        tokens = sentence[0]
        lemmas = []
        words = []
        ner = []
        pos = []
        for token in tokens:
            words.append(token[0].text)
            lemmas.append(token[1].text)
            pos.append(token[4].text)
            ner.append(token[5].text)
        parsing = sentence[1]

        kwargs = {"words": words,
                  "lemmas": lemmas,
                  "pos": pos,
                  "ner": ner}

        kwargs["dependencies"] = [None] * len(words)
        assert(sentence[2].tag == 'dependencies')
        for dependent in sentence[2]:
            assert(dependent.tag == 'dep')
            depType = dependent.attrib['type']
            governor = int(dependent[0].attrib['idx'])-1
            dependent = int(dependent[1].attrib['idx'])-1
            kwargs["dependencies"][dependent] = (depType, governor)
            
        if parse:
            kwargs["parse"] = parsing.text
                  
        return kwargs
        
    def iterSentences(self, parse=True):
        assert(self.root.tag == self.document)
        sentences = self.root[0] #self.root[0]
        assert(sentences.tag == 'sentences')
        for sentence in sentences:
            yield self.sentenceType(**self.sentenceExtractor(sentence, parse))
