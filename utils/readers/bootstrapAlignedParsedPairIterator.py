from __future__ import print_function

import gzip
import json
import collections

from altlex.utils.readers.parseMetadata import ParseMetadata
from altlex.utils.readers.alignedParsedPairIterator import AlignedParsedPairIterator

class BootstrapAlignedParsedPairIterator(AlignedParsedPairIterator):
    def __init__(self, indir, alignments, knownAltlexes, trainIndices=None,
                 verbose=False, combined=False):
        AlignedParsedPairIterator.__init__(self, indir, alignments, verbose, combined)
        self.knownAltlexes = knownAltlexes
        self.trainIndices = trainIndices
        
    def iterData(self, sentenceIndices=None,
                 datumIndices=None, modBy=10000):

        for sentenceId, datumId, label, altlex, pair in self.iterLabeledAltlexes(sentenceIndices,
                                                                                 datumIndices):

            if self.verbose and sentenceId % modBy == 0:
                print(sentenceId)

            if self.trainIndices and datumId in self.trainIndices:
                continue

            dp1 = ParseMetadata(pair[0]).datapoint(altlex[0][0], altlex[0][1])
            dp2 = ParseMetadata(pair[1]).datapoint(altlex[1][0], altlex[1][1])

            if tuple(dp1.getAltlexLemmasAndPos()) in self.knownAltlexes or tuple(dp2.getAltlexLemmasAndPos()) in self.knownAltlexes:
                yield sentenceId, datumId, dp1, label
                yield sentenceId, datumId, dp2, label
