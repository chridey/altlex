import os
import gzip
import json
import sys
import operator

import nltk

from altlex.utils import treeUtils
from altlex.utils import wordUtils

def getLemmas(lemmas):
    return reduce(operator.add, lemmas)

def getPossibleAltlexes(pairs, maxLen=7, verbose=False):
    #whitelist should be all discourse connectives
    #blacklist should be all modals and auxiliaries
    #for multi-sentence relations, allow the left siblings for the right sentence to be '0'

    ngrams = [[], []]
    for i in range(2):
        for s in range(len(pairs[i]['lemmas'])):
            lemmas = [j.lower() for j in getLemmas(pairs[i]['lemmas'][s:s+1])]
            try:
                parse = nltk.Tree.fromstring(pairs[i]['parse'][s])
            except IndexError:
                if verbose:
                    print ('Problem with different number of lemmas ({}) and parses ({}) {}:{}'.format(len(pairs[i]['lemmas']), len(pairs[i]['parse']), pairs[i]['lemmas'], pairs[i]['parse']))
                ngrams[i] = []
                break
            except ValueError:
                if verbose:
                    print('Problem parsing {}\n'.format(pairs[i]['parse'][s]))
                ngrams[i] = []
                break

            try:
                assert(len(lemmas) == len(parse.leaves()))
            except AssertionError:
                lemmas = [j for j in lemmas if j not in {'-lrb-', '-rrb-'}]
                try:
                    assert(len(lemmas) == len(parse.leaves()))
                except AssertionError:
                    if verbose:
                        print('Problem with match {}:{}'.format(lemmas, parse.leaves()))
                    ngrams[i] = []
                    break

            validLeftSiblings = ('V', 'N', 'S')
            if s > 0:
                validLeftSiblings = ('V', 'N', 'S', '0')

            ret = treeUtils.getConnectives(parse,
                                           maxLen,
                                           validLeftSiblings=frozenset(validLeftSiblings),
                                           blacklist = {tuple(k.split()) for k in wordUtils.modal_auxiliary},
                                           whitelist = wordUtils.all_markers,
                                           leaves = lemmas)
            
            pos = list(list(zip(*parse.pos()))[1])
            for t in ret:
                ngrams[i].append(lemmas[t[0]:t[1]] + pos[t[0]:t[1]])

    return ngrams


class ParsedPairIterator:
    def __init__(self, indir, verbose=False):
        assert(os.path.exists(indir))
        self.indir = indir
        self.verbose = verbose

    def __iter__(self):
        for filename in sorted(os.listdir(self.indir)):
            if self.verbose:
                print(filename)
            if not filename.endswith('.gz'):
                continue
            with gzip.open(os.path.join(self.indir, filename)) as f:
                j = json.load(f)

            assert(len(j) % 2 == 0)
            ret = []
            for index,sentence in enumerate(j):
                ret.append(sentence)
                if len(ret) == 2:
                    yield ret
                    ret = []
