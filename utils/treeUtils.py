import itertools

import nltk

from altlex.utils import wordUtils

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

    leftSiblings = []
    rightSiblings = []
    for i in range(len(leaves)):
        rightSiblings.append({i[:1] for i in getRightSiblings(i, parse) if i is not None})
        leftSiblings.append({i[:1] for i in getLeftSiblings(i, parse) if i is not None})
        
    #first go through the whitelist longest phrase to shortest and find any phrases that match and also satisfy
    #the valid right siblings
    foundIndices = set()
    for length in list(range(1, max(len(i) for i in whitelist)+1))[::-1]:
        #go through from longest to shortest
        #only add shorter phrases if they do not overlap with a longer phrase
        for i in range(len(leaves)-length):
            ls = leftSiblings[i]
            rs = rightSiblings[i+length]
            if ls & validLeftSiblings and rs & validRightSiblings and (tuple(leaves[i:i+length]) in whitelist) and not len(set(range(i,i+length)) & foundIndices):
                foundIndices.update(range(i,i+length))
                ret.append((i,i+length))

    #then find any phrases that match the patterns we selected
    for length in range(1, maxLen+1):
        for i in range(len(leaves)-length):
            #make sure they do not overlap with a known connective
            if len(set(range(i,i+length)) & foundIndices):
                continue
            
            ls = leftSiblings[i]
            rs = rightSiblings[i+length]
            phrase = leaves[i:i+length]

            if verbose:
                print(i, i+length, phrase, ls, rs)

            #make sure to ignore any parts of speech that we do not want in a connective
            if len(set(pos[i:i+length]) & invalid):
                continue

            #if this explicitly matches a connective we dont want ignore it
            if tuple(leaves[i:i+length]) in blacklist:
                continue

            if not len(ls & validLeftSiblings):
                continue
            if not len(rs & validRightSiblings):
                continue

            #finally, add if this contains an explicit connective or if it has a valid content word
            if any(set(item).issubset(leaves[i:i+length]) for item in whitelist) or \
            any (pos[x][:2] in content and (leaves[x],) not in blacklist for x in range(i, i+length)):
                foundIndices.update(range(i,i+length))
                ret.append((i,i+length))

    return ret


def getConnectives(parse,
                   maxLen=7,
                   content=frozenset(('VB', 'NN', 'RB', 'JJ')),
                   invalid=frozenset(('NNP', 'NNPS', 'SYM', ',', '(', ')', '"', "'", ';')),
                   validLeftSiblings=frozenset(('V', 'N', 'S')),
                   validRightSiblings=frozenset(('V', 'N', 'S')),                   
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
    
    leftSiblings = []
    rightSiblings = []
    for i in range(len(leaves)):
        rightSiblings.append({i[:1] for i in getRightSiblings(i, parse) if i is not None})
        leftSiblings.append({i[:1] for i in getLeftSiblings(i, parse) if i is not None})

    #first go through the whitelist longest phrase to shortest and find any phrases that match and also satisfy
    #the valid right siblings
    foundIndices = set()
    for length in list(range(1, max(len(i) for i in whitelist)+1))[::-1]:
        #go through from longest to shortest
        #only add shorter phrases if they do not overlap with a longer phrase
        for i in range(len(leaves)-length):
            if len(set(range(i,i+length)) & foundIndices):
                continue
            
            ls = leftSiblings[i]
            rs = rightSiblings[i+length]

            if not(len(ls & validLeftSiblings) and len(rs & validRightSiblings)):
                continue

            #allow adding preposition at beginning or end
            if (tuple(leaves[i:i+length]) in whitelist) or (tuple(leaves[i:i+length-1]) in whitelist and pos[i+length-1] in ('IN', 'TO')) or (tuple(leaves[i+1:i+length]) in whitelist and pos[i] in ('IN', 'TO')):
                foundIndices.update(range(i,i+length))
                ret.append((i,i+length))

    for length in range(1, maxLen+1):
        for i in range(len(leaves)-length):
            if len(set(range(i,i+length)) & foundIndices):
                continue

            ls = leftSiblings[i]
            rs = rightSiblings[i+length] 
            phrase = leaves[i:i+length]
            if verbose:
                print(i, i+length, phrase, ls, rs)

            if len(set(pos[i:i+length]) & invalid):
                continue
            if tuple(leaves[i:i+length]) in blacklist:
                continue

            if not(len(ls & validLeftSiblings) and len(rs & validRightSiblings)):
                continue

            #only allow one content word 
            totalContent = sum(pos[x][:2] in content and (leaves[x],) not in blacklist for x in range(i, i+length))
            if not totalContent:
                continue
                        
            ret.append((i,i+length))
            foundIndices.update(set((i,i+length)))
            
    return ret

def getParentNodes(index, parse):
    indices = parse.leaf_treeposition(index)
    return [parse[indices[:i+1]].label() for i in range(len(indices)-1)]

def extractParentNodes(phrase, parse):
    start = wordUtils.findPhrase(phrase, parse.leaves())
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
