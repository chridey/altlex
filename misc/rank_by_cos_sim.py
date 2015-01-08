#straightforward ranking, just take the causal altlexes and aggregate the cosine similarity (maybe averaged)

import sys
import math
from collections import defaultdict

weights = defaultdict(float)
counts = defaultdict(int)
nonweights = defaultdict(float)
noncounts = defaultdict(int)

if len(sys.argv) > 3:
    thresh = float(sys.argv[3])
else:
    thresh = 0.0
    
with open(sys.argv[1]) as f:
    for line in f:
        weight,altlex,*junk = line.split("\t")
        weight = float(weight)

        #split the altlex, treat named entities the same as prepositions
        al = altlex.split()
        r = []
        for a in al:
            if (a.isupper()):
                r.append('NP')
            else:
                r.append(a)
        altlex = ' '.join(r)
        
        if thresh == 0.0 or weight > thresh:
            weights[altlex] += weight
            counts[altlex] += 1
        else:
            nonweights[altlex] += weight
            noncounts[altlex] += 1

if sys.argv[2] == '0':
    for w in sorted (weights, key=weights.get):
        print(weights[w], counts[w], w)
elif sys.argv[2] in ('1', '2', '4'):
    if sys.argv[2] == '1':
        for w in sorted(weights, key=lambda x:math.log(counts[x],2)*1/(counts[x]+noncounts[x])*weights[x]/(nonweights[x]+1)):
            print(weights[w], nonweights[w], counts[w], noncounts[w], weights[w]/counts[w], math.log(counts[w],2)*1/(counts[w]+noncounts[w])*weights[w]/(nonweights[w]+1), w)
    elif sys.argv[2] == '2':
        for w in sorted(weights, key=lambda x:math.log(counts[x],2)*counts[x]/(counts[x]+noncounts[x])):
            print(weights[w], nonweights[w], counts[w], noncounts[w], weights[w]/counts[w], math.log(counts[w],2)*counts[w]/(counts[w]+noncounts[w]), w)
    else:
        for w in sorted(weights, key=lambda x:(counts[x]/(counts[x]+noncounts[x])-.5)*(weights[x]+nonweights[x])/(counts[x]+noncounts[x])*math.log(math.ceil(counts[x]/4),2)):
            print(weights[w], nonweights[w], counts[w], noncounts[w], weights[w]/counts[w], (counts[w]/(counts[w]+noncounts[w])-.5)*(weights[w]+nonweights[w])/(counts[w]+noncounts[w])*math.log(math.ceil(counts[w]/4),2), w)
elif sys.argv[1] == '3':
    for w in sorted (weights, key=lambda x:weights[x]/counts[x]):
        print(weights[w], counts[w], weights[w]/counts[w], w)
else:
    for w in sorted (weights, key=lambda x:weights[x]/counts[x]*math.log(counts[x])*2):
        print(weights[w], counts[w], weights[w]/counts[w]*math.log(counts[w])*2, w)
