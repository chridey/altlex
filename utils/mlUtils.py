import collections

def makeDeltaKLD(kldt, verbose=False):

    deltaKLD = collections.defaultdict(dict)

    for phraseType in kldt.keys():
        topKLD = kldt[phraseType][1].topKLD()
        for kld in topKLD:
            if kld[1] > kld[2]:
                score = kld[3]
            else:
                score = -kld[3]
            deltaKLD[phraseType][kld[0]] = score
    if verbose:
        for q in deltaKLD:
            print(q, len(deltaKLD[q]))

    return deltaKLD
