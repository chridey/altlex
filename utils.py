
def indexedSubset(x, i):
    '''takes in a list x and a collection of indices i and returns
    the values in x that are at those indices'''
    
    return list(zip(*filter(lambda x: x[0] in i, enumerate(x))))[1]
