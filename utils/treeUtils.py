import nltk

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

#TODO: given a phrase, return the outside and inside parses
#for now just assume the parse starts with the phrase
#TODO: find outside parse given any phrase in the sentence
def extractParse(phrase, parse, getSiblings=True):
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
                    if getSiblings == False:
                        return pos

    if not found:
        return None

    #now add anything with the same length that occurs after this phrase
    #sibling needs to occur at same level of tree but also have same parent node
    siblings.append(parse[sibling])
    for pos in parse.treepositions('preorder'): #subtrees():
        if pos > sibling and pos[:-1] == sibling[:-1]: # and len(pos) == len(sibling):
            if type(parse[pos]) == nltk.tree.Tree:
                siblings.append(parse[pos])
            
    return siblings

def extractRightSiblings(phrase, parse):
    siblings = extractParse(phrase, parse)
    if siblings is None:
        return []
    return [s.label() for s in siblings]

def _extractSelfCategoryIndex(phrase, parse):
    treeIndex = extractParse(phrase, parse, False)

    if treeIndex is None:
        print('cant find {} in {}'.format(phrase, parse))
        return None
    
    #iteratively check if parent node contains the entire phrase and only the phrase
    parentIndex = treeIndex[:-1]
    while len(parentIndex):
        if parse[parentIndex].leaves() == phrase:
            return parentIndex
        parentIndex = parentIndex[:-1]

    return None

def extractSelfParse(phrase, parse):
    try:
        return parse[_extractSelfCategoryIndex(phrase,parse)]
    except TypeError:
        return None

def extractSelfCategory(phrase, parse):
    catIndex = _extractSelfCategoryIndex(phrase,parse)
    if catIndex is None:
        return catIndex
    
    return parse[catIndex].label()
    
def extractParentCategory(phrase, parse):
    catIndex = _extractSelfCategoryIndex(phrase, parse)

    if catIndex is None:
        return catIndex

    return parse[catIndex[:-1]].label()
    
