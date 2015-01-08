import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer

DEBUG = 0

class Sentence:
    sn = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    
    def __init__(self, words, lemmas, pos, ner, parse=None):
        self.words = words
        self.lemmas = [self.wnl.lemmatize(i) for i in lemmas]
        self.stems = [self.sn.stem(i) for i in self.lemmas]
        self.pos = pos
        self.ner = ner
        self.parseString = parse
        if parse is not None:
            self.parse = nltk.Tree.fromstring(parse)
        else:
            self.parse = None

class SentenceReader:
    def __init__(self, xmlroot):
        self.root = xmlroot
        self.sentenceType = Sentence

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

        if parse:
            kwargs["parse"] = parsing.text
                  
        return kwargs
        
    def iterSentences(self, parse=True):
        assert(self.root.tag == 'document')
        sentences = self.root[0] #self.root[0]
        assert(sentences.tag == 'sentences')
        for sentence in sentences:
            yield self.sentenceType(**self.sentenceExtractor(sentence, parse))
