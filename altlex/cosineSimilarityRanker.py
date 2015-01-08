#straightforward ranking, just take the causal altlexes and aggregate the cosine similarity (maybe averaged)

import sys
import math
from collections import defaultdict

class CosineSimilarityRanker:
    def __init__(self, thresh=0.035):
        self.weights = defaultdict(float)
        self.counts = defaultdict(int)
        self.nonweights = defaultdict(float)
        self.noncounts = defaultdict(int)
        self.thresh = thresh

    def add(self, weight, altlex):
        if weight > self.thresh:
            self.weights[altlex] += weight
            self.counts[altlex] += 1
        else:
            self.nonweights[altlex] += weight
            self.noncounts[altlex] += 1

    def _iterByScore(self, scoreFunction):
        return ((i,scoreFunction(i)) for i in sorted(self.weights, key=lambda x:scoreFunction(x)))

    def iterByInverseCount(self):
        def score(i):
            return math.log(self.counts[i],2)*1/(self.counts[i]+self.noncounts[i])*self.weights[i]/(self.nonweights[i]+1)
        return self._iterByScore(score)

    def iterByWeightedCosSim(self):
        def score(x):
            return (self.counts[x]/(self.counts[x]+self.noncounts[x])-.5)*(self.weights[x]+self.nonweights[x])/(self.counts[x]+self.noncounts[x])*math.log(self.counts[x])
        return self._iterByScore(score)

    def iterByWeightedCosSimLogFreq(self):
        def score(x):
            return (self.counts[x]/(self.counts[x]+self.noncounts[x])-.5)*(self.weights[x]+self.nonweights[x])/(self.counts[x]+self.noncounts[x])*math.log(math.ceil(self.counts[x]/4),2)
        return self._iterByScore(score)

class TaggedCosineSimilarityRanker(CosineSimilarityRanker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal = defaultdict(int)
        self.noncausal = defaultdict(int)

    def evaluate(self, phrase, tag):
        if tag == 'causal':
            self.causal[phrase] += 1
        else:
            self.noncausal[phrase] += 1

if __name__ == '__main__':
    csr = CosineSimilarityRanker()
    
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

            csr.add(weight,altlex)
            
    if sys.argv[2] == '0':
        for phrase,score in csr.iterByWeight():
            print(phrase,score)
    elif sys.argv[2] == '1':
        for phrase,score in csr.iterByInverseCount():
            print(phrase,score)
    elif sys.argv[2] == '2':
        for phrase,score in csr.iterByWeightedCosSim():
            print(phrase,score)
