from nltk.tokenize import word_tokenize

from chnlp.utils.readers.sentenceReader import Sentence

def readAltlexes(altlexFile):
    with open(altlexFile) as f:
        altlexes = f.read().splitlines()

    for i in range(len(altlexes)):
        words = word_tokenize(altlexes[i])
        #do NP chunking
        altlexes[i] = Sentence(words, chunk=True)
        '''
        altlexes[i] = altlexes[i].replace("'s", " 's")
        altlexes[i] = altlexes[i].replace(":", " :")
        altlexes[i] = altlexes[i].replace(",", " ,")
        altlexes[i] = altlexes[i].replace('"', ' "')
        '''
        
    return altlexes

def matchAltlexes(sentence, altlexes):
    mx = 0
    argmax = None

    words = [l.lower() for l in sentence.words]
    for a in altlexes:
        alist = a.split()
        if words[:len(alist)] == alist:
            if len(alist) > mx:
                mx = len(alist)
                argmax = alist

    return argmax

def distantMatchAltlexes(sentence, altlexes):
    mx = 0
    argmax = None

    #first try just an anchored match (all words occur at some point)
    #then try POS based match

    #ie allow determiners to replace each other: That loss resulted vs the loss resulted
    #allow full NPs to replace lone determiners: That resulted vs That loss resulted
    #allow NPs to replace NPs: The loss resulted vs The housing market drop resulted
    #need to do NP chunking
    
    #also allow modifiers, prepositions, adjectives, adverbs, etc
    #some of these are taken care of by the chunking

    #essentially if we find a verb first, it is a different clause

    #1) if the altlex has a determiner/np, that can be replaced with any np
    #2) if the altlex has a verb, it must be replaced with the same verb
    #3) all other words must occur at some point

    words = []
    assert(sentence.parse is not None)
    try:
        assert(sentence.parse[0].label()[0] == 'S')
    except AssertionError:
        #print(sentence.parse)
        return None
    
    for t in sentence.parse[0]:
        f = t.flatten()
        if t.label()[0] == 'N':
            words.append(len(f))
            words.extend([0] * (len(f)-1))
        else:
            words.extend([0] * len(f))

    try:
        assert(len(words)==len(sentence.words))
    except AssertionError:
        print(words, sentence.words)
        print(sentence.parse)
        return None
    
    for a in altlexes:
        if len(a.lemmas) < 2:
            continue

        i = 0
        j = 0

        if 0:
            if a.lemmas[0] != sentence.lemmas[0]:
                continue
        
            while i < len(sentence.words) and j < len(a.words):
                if sentence.lemmas[i] == a.lemmas[j] and sentence.pos[i][0] == a.pos[j][0]:
                    j+=1
                i+=1
        else:
            #if sentence word/phrase is not part of an NP and is not a verb, ignore it
            #if altlex word/phrase is an NP, allow it to be replaced by any NP
            #if altlex word/phrase is a verb, the first verb must match (what about reporting? handle these later)
            #if altlex word is any other word, it must be in the sentence
            awords = []

            for t in a.parse:
                try:
                    f = t.flatten()
                except AttributeError:
                    awords.append(0)
                    continue

                #if t.label()[0] == 'N':
                awords.append(len(f))
                awords.extend([0] * (len(f)-1))

            assert(len(awords)==len(a.words))
                
            while i < len(words) and j < len(awords):
                #if current index has a true marker, its an NP, so it can be replaced by any NP
                if awords[j]:
                    if words[i]:
                        #we matched an NP, so increment the position in the altlex list
                        j += awords[j]
                else:
                    if sentence.lemmas[i] == a.lemmas[j] and sentence.pos[i][0] == a.pos[j][0]:
                        j+=1
                    elif sentence.pos[i][0].lower() == 'j' or sentence.pos[i][0].lower() == 'r':
                        pass
                    else: #edited to require exact matches sentence.pos[i][0].lower() == 'v':
                        #if we dont have a match, and the current sentence word is a verb
                        break
                        
                if words[i]:
                    i += words[i]
                else:
                    i += 1
            
        if j == len(a.words):
            print(a.words)
            print(a.pos)
            if len(a.words) > mx:
                mx = len(a.words)
                argmax = sentence.words[:i]

    return argmax
