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

class SentenceRelation(Sentence):
    def __init__(self, tag=None, **kwargs):
        super().__init__(**kwargs)
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
    
    if parse.label() == 'S' or parse.label() == 'SINV':
        #look for a noun phrase and a verb phrase
        np = False
        p = ''
        for tree in parse:
            if DEBUG:
                print("subtree S: " + tree.label(), tree.leaves(), p, preceding, current)
            #ignore punctuation
            #if not tree.label()[0].isalpha():
            #    continue
            #store noun phrase and anything else preceding
            if type(tree) == str:
                p += tree + ' '
            elif tree.label()[0] == 'S':
                return extractAltlex(tree, found, preceding + p, current)
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
            if DEBUG:
                print("subtree V: " + tree.label(), tree.leaves(), p, current)
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
            if DEBUG:
                print("SBAR subtree: " + tree.label(), tree.leaves())
            #if not tree.label()[0].isalpha():
            #    continue
            if type(tree) == str:
                p += tree + ' '
            elif tree.label()[0] == 'S' or tree.label() == 'PRN':
                return extractAltlex(tree, found, preceding + p + ' ')
            else:
                p += ' '.join(tree.leaves()) + ' '

    return extractAltlex(None, found, preceding, current)

#given a phrase, return the outside and inside parses
#for now just assume the parse starts with the phrase
#TODO: find outside parse given any phrase in the sentence
def extractParse(phrase, parse):
    assert(type(phrase) == list)
    siblings = []
    
    i = 0
    found = False
    sibling = None
    #subtrees does a preorder tree traversal
    for pos in parse.treepositions('preorder'): #subtrees():
        if type(parse[pos]) == nltk.tree.Tree:
            if found:
                sibling = pos
                break
            if parse[pos].height() == 2:
                leaves = parse[pos].leaves()
                if len(leaves) == 1 and leaves[0] == phrase[i]:
                    i += 1
                if i == len(phrase):
                    found = True

    if not found:
        return None

    #now add anything with the same length that occurs after this phrase
    siblings.append(parse[sibling])
    for pos in parse.treepositions('preorder'): #subtrees():
        if len(pos) == len(sibling) and pos > sibling:
            if type(parse[pos]) == nltk.tree.Tree:
                siblings.append(parse[pos])
            
    return siblings
        

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

class SentenceRelationReader(SentenceReader):
        def __init__(self, xmlroot):
            self.root = xmlroot
            self.sentenceType = SentenceRelation

        def sentenceExtractor(self, sentence, parse=True):
            kwargs = super().sentenceExtractor(sentence, parse)
            assert(sentence[2].tag == 'tag')
            kwargs["tag"] = sentence[2].text
            return kwargs
