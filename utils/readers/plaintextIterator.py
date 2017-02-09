from spacy.en import English
from nltk.stem import SnowballStemmer

from altlex.utils.dependencyUtils import tripleToList

class PlaintextIterator:
    nlp = English()
    stemmer = SnowballStemmer('english')
    
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            for line in f:                
                parsed_line = self.nlp(unicode(line.strip(), encoding='utf-8'))

                offset = 0
                for sentence in parsed_line.sents:
                    metadata = dict(deps=[],
                                    lemmas=[],
                                    pos=[],
                                    parse=[],
                                    words=[],
                                    ner=[],
                                    stems=[])
                    
                    for index,word in enumerate(sentence):
                        metadata['lemmas'].append(word.lemma_)
                        metadata['pos'].append(word.tag_)
                        metadata['ner'].append('O')
                        metadata['stems'].append(self.stemmer.stem(unicode(word)))
                        metadata['words'].append(unicode(word))

                        head = word.head.i-offset
                        if word.dep_ == 'ROOT':
                            head = -1
                        metadata['deps'].append((head, index, word.dep_))
                        
                    offset += len(sentence)

                    metadata['dependencies'] = tripleToList(metadata['deps'], len(metadata['words']))
                    yield metadata
