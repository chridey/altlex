def tripleToList(dependencies, length):
    '''take in a list of triples (gov, dep, rel)
    and return a list of doubles (rel, gov)
    where the dep is in sentence order'''
    ret = [None] * length
    for gov,dep,rel in dependencies:
        ret[dep] = (rel.lower(), gov)
    return ret

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
