import itertools

import nltk

#try replacing noun phrases with just generic NP
#right now this only does head noun but would need to recurse into other phrases to replace
#maybe should also stem verbs
#problem is that need to consider determiners separately (ie "that" should be treated differently since we know its a co-ref)

#really need to aggregate these somehow though
#try NER

DEBUG = 0

altlexPatterns = {
    4:
    [[('TO','IN'), ('DT','WDT','PRP$','PDT'), ('NN', 'NNS'),  ('TO','IN','WDT','WP','WRB')]],
    3:
    [[('TO','IN'), ('DT','WDT','PRP$','PDT'), ('NN', 'NNS')],
     [('TO','IN'), ('NN', 'NNS'),  ('TO','IN','WDT','WP','WRB')]],
    2:
    [[('TO','IN'), ('NN', 'NNS')],
     [('NN', 'NNS'),  ('TO','IN','WDT','WP','WRB')],
     [('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'), ('TO','IN','WDT','WP','WRB')],
     [('JJ', 'JJR', 'JJS'), ('TO','IN','WDT','WP','WRB')]],
    1:
    [[('NN', 'NNS')],
     [('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')],
     [('JJ', 'JJR', 'JJS')],
     [('RB', 'RBR', 'RBS')]]
}
altlexPatternSet = set(itertools.chain(*((itertools.product(*j)) for i in altlexPatterns for j in altlexPatterns[i])))

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

    if not found or sibling is None:
        return None

    #now add anything with the same length that occurs after this phrase
    #sibling needs to occur at same level of tree but also have same parent node
    
    siblings.append(parse[sibling])
    for pos in parse.treepositions('preorder'): #subtrees():
        if pos > sibling and pos[:-1] == sibling[:-1]: # and len(pos) == len(sibling):
            if type(parse[pos]) == nltk.tree.Tree:
                siblings.append(parse[pos])
            
    return siblings

'''
def extractRightSiblings(phrase, parse):
    siblings = extractParse(phrase, parse)
    if siblings is None:
        return []
    return [s.label() for s in siblings]
'''

def getRightSiblings(endIndex, parse):

    index = 0
    sibling = ['']
    origParse = parse
    #print(start, end)
    while index < len(origParse.leaves()):
        #print(index, parse.leaves(), sibling)
        if index >= endIndex:
            return sibling 
        #if endIndex >= index + len(parse.leaves()) and len(parse) == 1:
        #    return None
        #figure out which subtree to traverse
        for i in range(len(parse)):
            #print(i, index, len(parse[i].leaves()))

            if type(parse[i]) != nltk.tree.Tree:
                print('Problem with getRightSiblings at {},{}: {}'.format(index, i, origParse))
                return [None]
            
            #this means the end of phrase lies in this subtree
            if index + len(parse[i].leaves()) >= endIndex:
                #print(True, len(parse))
                if index + len(parse[i].leaves()) == endIndex and len(parse[i].leaves()) == 1:
                    index += len(parse[i].leaves())
                sibling = [p.label() for p in parse[i+1:]] if i < len(parse) - 1 else ['']
                parse = parse[i]
                break
            index += len(parse[i].leaves())
    return [None]

def getLeftSiblings(startIndex, parse):
    if startIndex == 0:
        return ['0']
    node = parse.leaf_treeposition(startIndex)[:-1]
    siblings = ['']
    while len(node) > 0 and parse[node].leaves()[0] == parse.leaves()[startIndex]:
        current = list(node)
        siblings = []
        for i in range(node[-1]):
            current[-1] = i
            siblings.append(parse[current].label())
        node = node[:-1]
    return siblings

def findPhrase(phrase, source):
    try:
        start = [source[i:i+len(phrase)] for i in range(len(source))].index(phrase)
    except ValueError:
        return None
    return start

'''
def getRightSiblings(index, parse):
    indices = parse.leaf_treeposition(index)

    if indices[-2] < len(parse[indices[:-2]]):
        ret = []
        current = list(indices[:-1])
        for i in range(indices[-2]+1, len(parse[indices[:-2]])):
            current[-1] = i
            print(current)
            ret += [parse[current].label()]
        return ret
    return ['']

'''
def extractRightSiblings(phrase, parse):
    start = findPhrase(phrase, parse.leaves())
    if start is None:
        return [None]
    return getRightSiblings(start + len(phrase),
                            parse)

def extractLeftSiblings(phrase, parse):
    pass

def getConnectivesPatterns(parse,
                           validRightSiblings=frozenset(('V', 'N', 'S')),                   
                           blacklist=(),
                           whitelist=(),
                           pos=None,
                           leaves=None,
                           verbose=False):

    ret = []
    if pos is None:
        pos = list(zip(*parse.pos()))[1]
    if leaves is None:
        leaves = parse.leaves()

    rightSiblings = []
    for i in range(len(leaves)):
        rightSiblings.append({i[:1] for i in getRightSiblings(i, parse) if i is not None})

    #first go through the whitelist and find any phrases that match and also satisfy
    #the valid right siblings
    foundIndices = set()
    for length in list(range(1, max(len(i) for i in whitelist)+1))[::-1]:
        #go through from longest to shortest
        #only add shorter phrases if they do not overlap with a longer phrase
        for i in range(len(leaves)-length):
            rs = rightSiblings[i+length]
            if rs & validRightSiblings and tuple(leaves[i:i+length]) in whitelist and not set(range(i,i+length)) & foundIndices:
                foundIndices.update(range(i,i+length))
                ret.append((i,i+length))
                
    #then go through the patterns longest to shortest
    #disallow any patterns that contain any phrases in the blacklist
    for length in list(range(1, max(altlexPatterns)+1))[::-1]:
        #go through from longest to shortest
        #only add shorter phrases if they do not overlap with a longer phrase
        for i in range(len(leaves)-length):
            rs = rightSiblings[i+length]
            if rs & validRightSiblings and tuple(pos[i:i+length]) in altlexPatternSet and not set(range(i,i+length)) & foundIndices and not set(leaves[i:i+length]) in blacklist:
                foundIndices.update(range(i,i+length))
                ret.append((i,i+length))

    return ret

def getConnectives2(parse,
                    maxLen=4,
                    content=('VB', 'NN', 'RB', 'JJ'),
                    invalid=frozenset(('NNP', 'NNPS', 'SYM', ',', '(', ')', '"', "'", ';')),
                    validRightSiblings=frozenset(('V', 'N', 'S')),
                    validLeftSiblings=frozenset(('V', 'N', 'S')),
                    blacklist=(),
                    whitelist=(),
                    pos=None,
                    leaves=None,
                    verbose=False):

    ret = []
    if pos is None:
        pos = list(zip(*parse.pos()))[1]
    if leaves is None:
        leaves = parse.leaves()

    leftSiblings = []
    rightSiblings = []
    for i in range(len(leaves)):
        rightSiblings.append({i[:1] for i in getRightSiblings(i, parse) if i is not None})
        leftSiblings.append({i[:1] for i in getLeftSiblings(i, parse) if i is not None})
        
    #first go through the whitelist and find any phrases that match and also satisfy
    #the valid right siblings
    foundIndices = set()
    for length in list(range(1, max(len(i) for i in whitelist)+1))[::-1]:
        #go through from longest to shortest
        #only add shorter phrases if they do not overlap with a longer phrase
        for i in range(len(leaves)-length):
            ls = leftSiblings[i]
            rs = rightSiblings[i+length]
            if ls & validLeftSiblings and rs & validRightSiblings and (tuple(leaves[i:i+length]) in whitelist or length == 1 and leaves[i] == 'so' and pos[i] == 'IN') and not set(range(i,i+length)) & foundIndices:
                foundIndices.update(range(i,i+length))
                ret.append((i,i+length))

    #then go through the patterns longest to shortest
    #disallow any patterns that contain any phrases in the blacklist
    for length in list(range(1, maxLen+1))[::-1]:
        #go through from longest to shortest
        #only add shorter phrases if they do not overlap with a longer phrase
        for i in range(len(leaves)-length):
            ls = leftSiblings[i]
            rs = rightSiblings[i+length]
            if rs & validRightSiblings and \
            ls & validLeftSiblings and \
            tuple(pos[i:i+length]) in altlexPatternSet and not set(range(i,i+length)) & foundIndices and not set(leaves[i:i+length]) in blacklist:
                foundIndices.update(range(i,i+length))
                ret.append((i,i+length))

    return ret

def getConnectives(parse,
                   maxLen=7,
                   content=('VB', 'NN', 'RB', 'JJ'),
                   invalid=('NNP', 'NNPS'), #, 'SYM', ',', '(', ')', '"', "'", ';'),
                   validLeftSiblings=('V', 'N', 'S'),
                   validRightSiblings=('V', 'N', 'S'),                   
                   blacklist=(),
                   whitelist=(),
                   pos=None,
                   leaves=None,
                   verbose=False):

    '''
    find the shortest non-subset phrases that satisfy all conditions

    blacklist - disallow these exact phrases (they also dont count as content words)
    whitelist - allow these exact phrases regardless of the P-O-S in content
    '''

    ret = []
    if pos is None:
        pos = list(zip(*parse.pos()))[1]
    if leaves is None:
        leaves = parse.leaves()
    invalid = set(invalid)
    
    leftSiblings = []
    rightSiblings = []
    for i in range(len(leaves)):
        leftSiblings.append(getLeftSiblings(i, parse))
        rightSiblings.append(getRightSiblings(i, parse))
        
    for length in range(1, maxLen+1):
        for i in range(len(leaves)-length):
            ls = leftSiblings[i] #getLeftSiblings(i, parse) #
            rs = rightSiblings[i+length] #getRightSiblings(i+length, parse) #
            phrase = leaves[i:i+length]
            if verbose:
                print(i, i+length, phrase, ls, rs)

            if not len(set(pos[i:i+length]) & invalid) and \
                   (any(set(item).issubset(leaves[i:i+length]) for item in whitelist) or \
               (not tuple(leaves[i:i+length]) in blacklist and 
            any (pos[x][:2] in content and (leaves[x],) not in blacklist for x in range(i, i+length)))) and \
            len(ls) and any(x and x[0] in validLeftSiblings for x in ls) and \
            len(rs) and any(x and x[0] in validRightSiblings for x in rs):

                if not any(findPhrase(leaves[p[0]:p[1]], phrase) is not None for p in ret):
                    ret.append((i,i+length))
                
    return ret

def getParentNodes(index, parse):
    indices = parse.leaf_treeposition(index)
    return [parse[indices[:i+1]].label() for i in range(len(indices)-1)]

def extractParentNodes(phrase, parse):
    start = findPhrase(phrase, parse.leaves())
    if start is None:
        return [None]

    return getParentNodes(start + len(phrase) - 1,
                          parse)
    
def _extractSelfCategoryIndex(phrase, parse):
    treeIndex = extractParse(phrase, parse, False)

    if treeIndex is None:
        if DEBUG:
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
    
def treesFromString(string):
    trees = []
    treeString = ''
    for line in string.split('\n'):
        if not line:
            continue
        if line.startswith('('):
            if treeString:
                trees.append(nltk.Tree.fromstring(treeString))
            treeString = ''
        treeString += line
    if treeString:
        trees.append(nltk.Tree.fromstring(treeString))
    return trees
