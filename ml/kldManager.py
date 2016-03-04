import os
import collections

from altlex.ml.tfkld import KldTransformer

class KLDManager:
    def __init__(self,
                 kldDir=os.path.join(os.path.join(os.path.dirname(__file__),
                                                  '..'),
                                     'config'),
                 kldt=None,
                 suffix='.kldt',
                 verbose=False):

        self.verbose = verbose
        self.suffix = suffix
        if kldt is not None:
            self.kldt = kldt
            self.kldDir = None
        else:
            self.kldDir = kldDir
            self.loadKLD()

        self.deltaKLD = None

    def score(self, ngram, pos):
        if self.deltaKLD is None:
            self.makeDeltaKLD()
            
        scores = {}
        for key in self.deltaKLD:
            scores[key] = self.deltaKLD[key].get(tuple(ngram + pos), 0)
        return scores

    def loadKLD(self):
        self.kldt = {}
        for filename in os.listdir(self.kldDir):
            if filename.endswith(self.suffix):
                if self.verbose:
                    print('loading {}...'.format(filename))
            
                kldtType = filename[:len(filename)-len(self.suffix)]
                self.kldt[kldtType] = KldTransformer.load(os.path.join(self.kldDir,
                                                                       filename))

    def makeDeltaKLD(self):

        self.deltaKLD = collections.defaultdict(dict)

        for phraseType in self.kldt.keys():
            topKLD = self.kldt[phraseType].topKLD()
            for kld in topKLD:
                if kld[1] > kld[2]:
                    score = kld[3]
                else:
                    score = -kld[3]
                self.deltaKLD[phraseType][kld[0]] = score

        if self.verbose:
            for q in self.deltaKLD:
                print(q, len(self.deltaKLD[q]))
                
