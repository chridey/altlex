from chnlp.utils.readers.sentenceReader import SentenceReader,Sentence
    
class FastSentenceReader:
    def __init__(self, document):
        '''document is a list of strings in XML'''
        self.sentenceType = Sentence
        self.document = document
        
    def iterSentences(self, parse=True):
        inSentences = False
        inSentence = False
        inTokens = False
        inToken = False
        inDependencies = False
        
        kwargs = {"words": [],
                  "lemmas": [],
                  "pos": [],
                  "ner": []}
        for line in self.document:
            if '<sentences>' in line:
                inSentences = True
            elif '</sentences>' in line:
                break
            elif '<sentence' in line:
                inSentence = True
            elif '</sentence' in line:
                #print(kwargs)
                yield self.sentenceType(**kwargs)
                inSentence = False
                kwargs = {"words": [],
                          "lemmas": [],
                          "pos": [],
                          "ner": []}
            elif inSentence:
                if '<parse>' in line and '</parse>'  in line:
                    if parse:
                        kwargs['parse'] = line.replace('<parse>','').replace('</parse>','').strip()
                elif '<tokens>' in line:
                    inTokens = True
                elif '</tokens>' in line:
                    inTokens = False
                elif inTokens:
                    if '<token' in line:
                        inToken = True
                    elif '</token>' in line:
                        inToken = False
                    elif inToken:
                        for wordType in 'word','lemma','POS', 'NER':
                            tag = '<{}>'.format(wordType)
                            if tag in line:
                                endTag = '</{}>'.format(wordType)
                                if wordType in {'word','lemma'}:
                                    wordType += 's'
                                kwargs[wordType.lower()].append(line.replace(tag,'').replace(endTag,'').strip())
                                break
                elif '<dependencies' in line:
                    kwargs['dependencies'] = ''
                    inDependencies = True
                elif '</dependencies>' in line:
                    inDependencies = False
                elif inDependencies:
                    kwargs['dependencies'] += line
