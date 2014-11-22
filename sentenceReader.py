import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer

class Sentence:
    sn = SnowballStemmer("english")
    
    def __init__(self, words, lemmas, pos, ner, parse=None):
        self.words = words
        self.lemmas = lemmas
        self.stems = [self.sn.stem(i) for i in self.lemmas]
        self.pos = pos
        self.ner = ner
        self.parseString = parse
        if parse is not None:
            self.parse = nltk.Tree.fromstring(parse)
        else:
            self.parse = None

class SentenceRelation(Sentence):
    def __init__(self, *args, **kwargs, tag=None):
        super().__init__(args, kwargs)
        self.tag = tag

#try replacing noun phrases with just generic NP
#right now this only does head noun but would need to recurse into other phrases to replace
#maybe should also stem verbs
#problem is that need to consider determiners separately (ie "that" should be treated differently since we know its a co-ref)

#really need to aggregate these somehow though
#try NER

def extractAltlex(parse, found=False, preceding='', current=''):
    #need way of breaking out of recursion and returning result
    if parse is None or type(parse) == str:
        if found == True:
            return preceding
        else:
            return ''

    #print(parse.label())
        
    #first find NP and VP
    #if none, dont bother        
    if parse.label() == 'ROOT':
        return extractAltlex(parse[0], found, preceding, current)
    
    if parse.label() == 'S':
        #look for a noun phrase and a verb phrase
        np = False
        p = ''
        for tree in parse:
            #ignore punctuation
            #if not tree.label()[0].isalpha():
            #    continue
            #store noun phrase and anything else preceding
            if type(tree) == str:
                p += tree + ' '
            elif tree.label()[0] == 'N':
                np = True
                #try replacing noun phrases with just generic NP
                #p += ' NP '
                p += ' '.join(tree.leaves()) + ' '
            elif tree.label()[0] == 'V':
                if np == False:
                    return extractAltlex(None, found, preceding, current)
                else:
                    return extractAltlex(tree, found, preceding, p + ' ')
            else:
                p += ' '.join(tree.leaves()) + ' '
                
    if parse.label()[0] == 'V':
        vp = False
        p = ''
        for tree in parse:
            #only care about verbs that take sentences as objects
            if type(tree) == str:
                p += tree + ' '
            elif tree.label()[0] == 'V':
                vp = True
                p += ' '.join(tree.leaves()) + ' '
            #should maybe change this to make it a little more restrictive
            #ie determine that the sentence is a direct object instead of an adjoining clause
            #maybe i should just allow PPs and ADVPs in between
            elif tree.label()[0] == 'S':
                #dont know why there wouldnt be a verb in a verb phrase
                if vp == False:
                    return extractAltlex(None, found, preceding, current)
                else:
                    return extractAltlex(tree, True, preceding + current + p + ' ')
            elif vp is True:
                if tree.label()[0] == 'P' or tree.label()[0] == 'A':
                    p += ' '.join(tree.leaves()) + ' '
                else:
                    return extractAltlex(None, found, preceding, current)
            else:
                p += ' '.join(tree.leaves()) + ' '
                
    if parse.label() == 'SBAR':
        #return everything before the S
        p = ''
        for tree in parse:
            #if not tree.label()[0].isalpha():
            #    continue
            if type(tree) == str:
                p += tree + ' '
            elif tree.label()[0] == 'S':
                return extractAltlex(tree, found, preceding + p + ' ')
            else:
                p += ' '.join(tree.leaves()) + ' '

    return extractAltlex(None, found, preceding, current)

class SentenceReader:
    def __init__(self, xmlroot):
        self.root = xmlroot

    def iterSentences(self, parse=True):
        sentences = self.root[0][0]
        assert(sentences.tag == 'sentences')
        for sentence in sentences:
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
            if parse:
                yield Sentence(words, lemmas, pos, ner, parsing.text)
            else:
                yield Sentence(words, lemmas, pos, ner)
