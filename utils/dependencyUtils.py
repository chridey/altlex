import collections

def tripleToList(dependencies, length, multipleParents=False, ignoreOutOfBounds=False):
    '''take in a list of triples (gov, dep, rel)
    and return a list of doubles (rel, gov)
    where the dep is in sentence order'''
    ret = [None] * length
    for gov,dep,rel in dependencies:
        if dep >= length and ignoreOutOfBounds:
            continue
        if multipleParents:
            if ret[dep] is None:
                ret[dep] = []
            ret[dep].append((rel.lower(), gov))
        else:
            ret[dep] = [rel.lower(), gov]
    return ret

    #TODO: handle cases with more than one parent

def combineDependencies(*dependenciesList):
    offset = 0
    combined = []
    for dependencies in dependenciesList:
        for dependency in dependencies:
            if dependency is None:
                combined.append(None)
            else:
                combined.append((dependency[0], dependency[1]+offset))
        offset += len(dependencies)
    return combined

def combineDependenciesMultiple(*dependenciesList):
    offset = 0
    combined = []
    for dependencies in dependenciesList:
        for dependency in dependencies:
            if dependency is None:
                combined.append(None)
            else:
                appendee = []
                for parent in dependency:
                    appendee.append((parent[0], parent[1]+offset))
                combined.append(appendee)
        offset += len(dependencies)
    return combined

def splitDependencies(dependencies, connectiveIndices):
    '''split the dependency tree into two trees,
    removing any links that cross the connective
    and adding new root nodes as necessary
    '''
    start, end = connectiveIndices
    altlexIndices = set(range(start, end))
    newDependencies = {'prev': [None]*start,
                       'altlex': [None]*(end-start),
                       'curr': [None]*(len(dependencies)-end)}
    for dep,rel_gov in enumerate(dependencies):
        if rel_gov is None:
            continue
        rel,gov = rel_gov
        #if the relationship crosses a barrier, remove it
        if gov == -1:
            if dep < start:
                newDependencies['prev'][dep] = rel,gov
            elif dep >= end:
                newDependencies['curr'][dep-end] = rel,gov
            elif dep in altlexIndices:
                newDependencies['altlex'][dep-start] = rel,gov
        elif dep < start and gov < start:
            newDependencies['prev'][dep] = rel,gov
        elif dep >= end and gov >= end:
            newDependencies['curr'][dep-end] = rel,gov-end
        elif dep in altlexIndices and gov in altlexIndices:
            newDependencies['altlex'][dep-start] = rel,gov-start

    #finally, make sure all the new dependencies have a root node

    #print(newDependencies)
    
    for section in newDependencies:
        if any(i == ('root', -1) for i in newDependencies[section]):
            continue
        root = 0
        for dependency in newDependencies[section]:
            if dependency is not None:
                #if dependency[1] >= len(newDependencies[section]):
                #    continue
                if dependency[-1] != -1 and newDependencies[section][dependency[1]] is None:
                    #if root < len(newDependencies[section]):
                    root = dependency[1]
                    newDependencies[section][root] = 'root',-1 #UNDO
        if len(newDependencies[section]) and newDependencies[section][root] is None:
            newDependencies[section][root] = 'root',-1
    return newDependencies

def getRoot(parse):
    for i in range(len(parse)):
        if parse[i] is not None and parse[i][0] == 'root':
            return i
    return None
                    
def getRootMultiple(parse):
    for i in range(len(parse)):
        if parse[i] is not None and len(parse[i]) == 1 and parse[i][0][0] == 'root':
            return i
    return None

def iterDependencies(parse):
    '''iterate over the dependency structure from
    the leaves to the root
    (useful for dependency embeddings)'''
    pass

def getEventAndArguments(parse):
    '''take in a dependency parse and return the
    main event (usually a verb) and any corresponding
    (noun) arguments'''
    root = getRoot(parse)
    if root is None:
        return None,[]
    arguments = []
    for i in range(len(parse)):
        if parse[i] is not None and parse[i][1] == root:
            arguments.append(i)
    return root,arguments
                                            
def getEventAndArgumentsMultiple(parse):
    '''take in a dependency parse and return the
    main event (usually a verb) and any corresponding
    (noun) arguments'''
    root = getRoot(parse)
    if root is None:
        return None,[]
    arguments = []
    for i in range(len(parse)):
        if parse[i] is not None and any(j[1] == root for j in parse[i]):
            arguments.append(i)
    return root,arguments

#modify a dependency parse so that compounds/names/mwes are combined
def getCompounds(parse):
    ret = collections.defaultdict(list)
    for i in range(len(parse)):
        if parse[i] is None:
            continue
        if type(parse[i][0]) in (str, unicode):
            if parse[i][0] in ('compound', 'name', 'mwe'):
                ret[parse[i][1]].append(i)
        else:
            for j in parse[i]:
                if j[0] in ('compound', 'name', 'mwe'):
                    ret[j[1]].append(i)
    return ret

def getAllEventsAndArguments(parse):
    '''given a dependency parse return a list of all
    events (verbs) and their nsubj,nsubjpass,dobj,and iobj'''

    #use subjects and objects to find the predicates
    #0 is rel, 1 is gov
    ret = collections.defaultdict(dict)
    for i in range(len(parse)):
        if parse[i] is None:
            print('Problem with parse: {}'.format(parse))
            continue
        if type(parse[i][0]) in (str, unicode):
            if parse[i][0] in ('nsubj', 'nsubjpass', 'dobj', 'iobj'):
                if parse[i][0] not in ret[parse[i][1]]:
                    ret[parse[i][1]][parse[i][0]] = []

                ret[parse[i][1]][parse[i][0]].append(i)
        else:
            for j in parse[i]:
                if j[0] in ('nsubj', 'nsubjpass', 'dobj', 'iobj'):
                    if j[0] not in ret[j[1]]:
                        ret[j[1]][j[0]] = []

                    ret[j[1]][j[0]].append(i)

    return ret
                                                            
def makeDependencies(words, deps):
    #make the root of the altlex the new root
    #for every edge on the path from the original root to the new root flip the direction
    altlexStart = len(words[0])
    altlexEnd = len(words[0]+words[1])
    altlexRoot = None

    #print()
    #print(datum['altlex'], altlexStart, altlexEnd, words[altlexStart:altlexEnd])
    #print(zip(words, deps))
    
    if len(words[1])==1:
        altlexRoot = altlexStart
    else:
        for i in range(altlexStart, altlexEnd):
            rel,gov = deps[i]
            #if the gov is the parent of the altlex root or there is none
            if gov in range(altlexStart, altlexEnd) and (altlexRoot is None or i == altlexRoot):
                altlexRoot = gov

    if altlexRoot is None:
        
        for i in range(altlexStart, altlexEnd):
            if deps[i][0] != 'det':
                altlexRoot = i
                break
        
    assert(altlexRoot is not None)

        
    parent = -1
    rel = 'root'
    child = altlexRoot
    while child != -1:
        dep, gov = deps[child]
        deps[child][0] = rel
        deps[child][1] = parent

        parent = child
        child = gov
        rel = dep + '_rev'

    #print(zip(words, deps))

    final_words = words[0] + words[1] + words[2]
    d = [('ROOT', None, None)]
    for i,j in enumerate(deps):
        if j is None:
            d.append((None,None,None))
        else:
            d.append((final_words[i].lower(), j[0], j[1]+1))

    return d
