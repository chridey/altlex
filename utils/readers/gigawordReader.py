import os
import gzip
import time

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer

class GigawordReader:
    def __init__(self, gigaword_dir):
        self.gigaword_dir = gigaword_dir

    def iterFiles(self):
        dirs = [self.gigaword_dir]
        #recurse through directory
        while len(dirs):
            curr_dir = dirs.pop()
            for t in os.listdir(curr_dir):
                fullPath = os.path.join(curr_dir,t)
                #if t is a directory, add to dirs
                if os.path.isdir(t):
                    dirs.append(fullPath)
                else:
                    if not fullPath.endswith('.gz'):
                        continue
                    print(fullPath)
                    print(time.time())
                    with gzip.open(fullPath, "rb") as gf:
                        inDoc = False
                        inPara = False
                        for line in gf:
                            #line = str(line)
                            if line.startswith(b'<DOC'):
                                assert(inDoc == False)
                                inDoc = True
                                sentences = []
                            elif line.startswith(b'</DOC>'):
                                assert(inDoc == True)
                                inDoc = False
                                yield sentences
                            elif line.startswith(b'<P>'):
                                assert(inPara == False)
                                inPara = True
                                currSentence = b''
                            elif line.startswith(b'</P>'):
                                assert(inPara == True)
                                inPara = False
                                try:
                                    sentences.append(currSentence.decode())
                                except Exception as e:
                                    print(e)
                            elif inPara:
                                if len(currSentence):
                                    currSentence += b' '

                                currSentence += line.replace(b'\n', b'')
                                
                                    
                        gf.close()

#standard unformatted sentence text reader 
class GigawordSentenceReader:
    wnl = WordNetLemmatizer()
    sn = SnowballStemmer('english')

    def __init__(self, document, lower=True):
        self.document = document
        self.lower = lower
        
    def iterSentences(self):
        for sentence in self.document:
            words = word_tokenize(sentence)
            if self.lower:
                words = [i.lower() for i in words]
            yield GigawordSentence(words)

class GigawordSentenceReaderLemmatized(GigawordSentenceReader):
    def iterSentences(self):
        for sentence in self.document:
            words = word_tokenize(sentence)
            if self.lower:
                words = [i.lower() for i in words]
            lemmas = []
            for w in words:
                l = self.wnl.lemmatize(w)
                if l == w:
                    l = self.wnl.lemmatize(w, 'v')
                lemmas.append(l)

            yield GigawordSentence(lemmas)

class GigawordSentenceReaderStemmed(GigawordSentenceReader):
    def iterSentences(self):
        for sentence in self.document:
            words = word_tokenize(sentence)
            if self.lower:
                words = [i.lower() for i in words]
            stems = []
            for w in words:
                l = self.sn.stem(w)
                stems.append(l)

            yield GigawordSentence(stems)

class GigawordSentence:
    
    def __init__(self, words, lemmatize=True, stem=False):
        self.words = words
        self.nelemmas = words

