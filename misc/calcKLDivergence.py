#for each file, create counts of words appearing for that connective and then not on both sides

import os
import sys
import collections
import math
import itertools
import multiprocessing
import operator

from chnlp.utils.utils import makeNgrams, balance
from chnlp.misc import wikipedia, wiknet
from chnlp.ml.tfkld import KldTransformer, TfkldFactorizer

from chnlp.utils import wordUtils
from chnlp.utils.treeUtils import extractRightSiblings, extractParentNodes, getConnectives, getConnectives2

import nltk

from sklearn.linear_model import SGDClassifier

stemmer = nltk.PorterStemmer()

import gzip
import json

def readPhrasePairs(filename, stem=True, outfile=None, phrases=None, maxLen=0):
    if phrases is None:
        phrases = collections.defaultdict(dict)
    else:
        phrases = phrases
        
    maxLen = maxLen
    with gzip.open(filename) as f:
        for index,line in enumerate(f):
            if index % 10000 == 0 and index > 0:                
                print(index)
                #break
            #we only care about the ones that are different
            try:
                phrase1, phrase2, scores, alignments, ints, junk = line.decode('utf-8').split('|||')
            except ValueError:
                print('Problem with {}'.format(line))
                continue
            if phrase1 != phrase2:
                if stem:
                    phrase1 = tuple(stemmer.stem(i) for i in phrase1.split())
                    phrase2 = tuple(stemmer.stem(i) for i in phrase2.split())
                else:
                    phrase1 = tuple(phrase1.split())
                    phrase2 = tuple(phrase2.split())
                if len(phrase1) > maxLen:
                    maxLen = len(phrase1)
                if len(phrase2) > maxLen:
                    maxLen = len(phrase2)
                    
                phrases[' '.join(phrase1)][' '.join(phrase2)] = scores.split()#.add(phrase2) #
                phrases[' '.join(phrase2)][' '.join(phrase1)] = scores.split()#.add(phrase1) #

    print(len(phrases))
    print(maxLen)
    if outfile is not None:
        with gzip.open(outfile, 'w') as f:
            json.dump({'phrases': phrases, 'maxLen': maxLen}, f)
    return phrases, maxLen

class PairedFileIterator:
    #format is <sent1>\t<sent2>\t<score>
    def __init__(self, infile, metadata, verbose=False):
        self.infile = infile

    def __iter__(self):
        with open(self.infile) as f:
            for line in f:
                try:                
                    s1, s2, score = line.decode('utf-8').split(splitToken)
                except ValueError:
                    if verbose:
                        print('Problem with line {}'.format(line))
                    continue

                words1 = nltk.word_tokenize(s1)
                words2 = nltk.word_tokenize(s2)

                w1 = ' '.join(words1)
                w2 = ' '.join(words2)

                if w1 not in metadata or w2 not in metadata:
                    if verbose:
                        print('Problem with {}\t{} not in metadata'.format(w1.encode('utf-8'),
                                                                           w2.encode('utf-8')))
                    continue

                return metadata[w1], metadata[w2]

class ParsedPairIterator:
    def __init__(self, indir, verbose=False):
        assert(os.path.exists(indir))
        self.indir = indir
        self.verbose = verbose

    def __iter__(self):
        for filename in os.listdir(self.indir):
            if self.verbose:
                print(filename)
            with gzip.open(os.path.join(self.indir, filename)) as f:
                j = json.load(f)

            assert(len(j[0]['sentences'][0]) % 2 == 0)
            ret = []
            for index,sentence in enumerate(j[0]['sentences'][0]):
                ret.append(sentence)
                if len(ret) == 2:
                    yield ret
                    ret = []

def getDataNgrams(iterator, phrases, maxLen):
    data = []

    for pairs in iterator:
        ngrams = [[], []]
        parseWords1 = [i.lower() for i in wiknet.getLemmas(pairs[0]['words'])]
        parseWords2 = [i.lower() for i in wiknet.getLemmas(pairs[1]['words'])]

        lemmas1 = [i.lower() for i in wiknet.getLemmas(pairs[0]['lemmas'])]
        lemmas2 = [i.lower() for i in wiknet.getLemmas(pairs[1]['lemmas'])]
        stems1 = [stemmer.stem(i) for i in parseWords1]
        stems2 = [stemmer.stem(i) for i in parseWords2]
        pos1 = wiknet.getLemmas(pairs[0]['pos'])
        pos2 = wiknet.getLemmas(pairs[1]['pos'])

        try:
            parses1 = [nltk.Tree.fromstring(i) for i in pairs[0]['parse']]
            parses2 = [nltk.Tree.fromstring(i) for i in pairs[1]['parse']]
        except ValueError:
            print('Problem parsing {}\n{}\n'.format(pairs[0]['parse'],
                                                    pairs[1]['parse']))
            continue

        for i in range(maxLen):
            for j in range(len(lemmas1)-maxLen):
                if ' '.join(lemmas1[j:j+i]) in phrases:
                    siblings = reduce(operator.add,
                                      [extractRightSiblings(parseWords1[j:j+i],
                                                            p) for p in parses1])
                    parents = set(map(lambda x:x[0],
                                      filter(None,
                                             reduce(operator.add,
                                                    [extractParentNodes(parseWords1[j:j+i],
                                                                        p) for p in parses1]))))
                    for sibling in set(filter(None, siblings)):
                        ngrams[0].append(lemmas1[j:j+i] + [k[0] for k in pos1[j:j+i]] + ['SIBLING_{}'.format(sibling)] + ['PARENT_VP_{}'.format('V' in parents)])

        for i in range(maxLen):
            for j in range(len(lemmas2)-maxLen):
                if ' '.join(lemmas2[j:j+i]) in phrases:
                    siblings = reduce(operator.add,
                                      [extractRightSiblings(parseWords2[j:j+i],
                                                            p) for p in parses2])
                    parents = set(map(lambda x:x[0],
                                      filter(None,
                                             reduce(operator.add,
                                                    [extractParentNodes(parseWords2[j:j+i],
                                                                        p) for p in parses2]))))
                    for sibling in set(filter(None, siblings)):
                        ngrams[1].append(lemmas2[j:j+i] + [k[0] for k in pos2[j:j+i]] + ['SIBLING_{}'.format(sibling)] + ['PARENT_VP_{}'.format('V' in parents)])

        data.append(ngrams)

    return data

#whitelist should be all discourse connectives
#blacklist should be all modals and auxiliaries
#for multi-sentence relations, allow the left siblings for the right sentence to be '0'
def getNgrams(pairs, maxLen=7):
    ngrams = [[], []]
    for i in range(2):
        for s in range(len(pairs[i]['lemmas'])):
            lemmas = [j.lower() for j in wiknet.getLemmas(pairs[i]['lemmas'][s:s+1])]
            try:
                parse = nltk.Tree.fromstring(pairs[i]['parse'][s])
            except IndexError:
                print ('Problem with different number of lemmas ({}) and parses ({}) {}:{}'.format(len(pairs[i]['lemmas']), len(pairs[i]['parse']), pairs[i]['lemmas'], pairs[i]['parse']))
                ngrams[i] = []
                break
            except ValueError:
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
                    print('Problem with match {}:{}'.format(lemmas, parse.leaves()))
                    ngrams[i] = []
                    break

            validLeftSiblings = ('V', 'N', 'S')
            if s > 0:
                validLeftSiblings = ('V', 'N', 'S', '0')

            '''
            ret = getConnectives(parse,
                                 maxLen,
                                 validLeftSiblings=validLeftSiblings,
                                 blacklist = {tuple(k.split()) for k in wordUtils.modal_auxiliary},
                                 whitelist = wordUtils.all_markers,
                                 leaves = lemmas)
            '''
            
            ret = getConnectives2(parse,
                                  validLeftSiblings=validLeftSiblings,
                                  blacklist = {tuple(k.split()) for k in wordUtils.modal_auxiliary},
                                  whitelist = wordUtils.all_markers,
                                  leaves = lemmas)
            
            pos = list(list(zip(*parse.pos()))[1])
            for t in ret:
                ngrams[i].append(lemmas[t[0]:t[1]] + pos[t[0]:t[1]])

    return ngrams

def getDataNgrams2(iterator, maxLen):
    data = []

    for pairs in iterator:
        ngrams = getNgrams(pairs, maxLen)
        if len(ngrams[0]) and len(ngrams[1]):    
            data.append(ngrams)

    return data

def calcNotInS1(counts, allCounts):
    p_num = {}
    p_denom = {}
    q_num = {}
    q_denom = {}
        
    for stem in counts['in_s2_not_in_s1']:
        p_num[stem] = counts['in_s2_not_in_s1'][stem]
        p_denom[stem] = counts['total'] - counts['in_s1'].get(stem, 0)

        q_num[stem] = allCounts['in_s2_not_in_s1'][stem] - counts['in_s2_not_in_s1'][stem]
        #just added -counts, make sure we dont subtract the connective and s1 twice
        q_denom[stem] = allCounts['total'] - allCounts['in_s1'].get(stem, 0) - counts['total'] + counts['in_s1'].get(stem, 0)

    return p_num, p_denom, q_num, q_denom

def calcInS1(counts, allCounts):
    p_num = {}
    p_denom = {}
    q_num = {}
    q_denom = {}
        
    for stem in counts['in_s2']:
        #percentage of sentences that have this feature given this connective
        p_num[stem] = counts['in_s2'][stem]
        p_denom[stem] = counts['total']
        #percentage of sentences that have this feature given not this connective
        q_num[stem] = allCounts['in_s2'][stem] - counts['in_s2'][stem]
        q_denom[stem] = allCounts['total'] - counts['total']
    
    return p_num, p_denom, q_num, q_denom

class DataFileIterator:
    def __init__(self, dirname, split_sent=False, word_pairs=False, max_ngrams=2, location=0):
        self.dirname = dirname
        self.split_sent = split_sent
        self.word_pairs = word_pairs
        self.max_ngrams = max_ngrams
        self.location = location

    def __iter__(self):
        print(self.split_sent, self.word_pairs, self.max_ngrams, self.location)

        discourse = wikipedia.loadDiscourse()

        for filename in os.listdir(self.dirname):
            if filename.startswith('.'):
                continue
            with open(filename) as f:
                for line in f:
                    try:                
                        s1, s2, score = line.strip().decode('utf-8').split('\t')
                    except ValueError:
                        print('Problem with line {}'.format(line.decode('utf-8')))
                        continue

                    if self.split_sent:
                        clause1, relation, clause2 = wikipedia.splitOnDiscourse(s1, discourse)
                        stems11 = ['1_' + stemmer.stem(i.lower()) for i in clause1]
                        stems12 = ['2_' + stemmer.stem(i.lower()) for i in clause2]

                        sents = nltk.sent_tokenize(s2)
                        if len(sents) == 2:
                            sent1 = nltk.word_tokenize(sents[0])
                            sent2 = nltk.word_tokenize(sents[1])
                        else:
                            #heuristically split the sentence
                            splitPct = 1. * len(stems11) / (len(stems11) + len(stems12))
                            words = nltk.word_tokenize(s2)
                            splitPoint = int(splitPct * len(words))
                            sent1 = words[:splitPoint+1]
                            sent2 = words[splitPoint+1:]

                        stems21 = ['1_' + stemmer.stem(i.lower()) for i in sent1]
                        stems22 = ['2_' + stemmer.stem(i.lower()) for i in sent2]

                        stems1 = makeNgrams(stems11, self.max_ngrams, self.location)
                        stems1.update(makeNgrams(stems12, self.max_ngrams, self.location))
                        stems2 = makeNgrams(stems21, self.max_ngrams, self.location)
                        stems2.update(makeNgrams(stems22, self.max_ngrams, self.location))

                        if self.word_pairs:
                            stems1 |= set(itertools.product(stems11, stems12))
                            stems2 |= set(itertools.product(stems21, stems22))
                    else:
                        stems = [stemmer.stem(i.lower()) for i in nltk.word_tokenize(s1)]
                        stems1 = makeNgrams(stems, self.max_ngrams)
                        if self.word_pairs:
                            stems1 |= set(itertools.permutations(stems, 2))

                        stems = [stemmer.stem(i.lower()) for i in nltk.word_tokenize(s2)]
                        stems2 = makeNgrams(stems, self.max_ngrams)
                        if self.word_pairs:
                            stems2 |= set(itertools.permutations(stems, 2))

            yield filename,stems1,stems2

class PreprocessedIterator:
    def __init__(self, phrases, seedSet):
        self.phrases = phrases
        self.seedSet = seedSet

    def __iter__(self):
        for datapoint in self.phrases:
            stems1 = set(tuple(i) for i in datapoint[0])
            stems2 = set(tuple(i) for i in datapoint[1])
            hasCausalIn1 = any(set(seed).issubset(stem) for stem in stems1 for seed in self.seedSet['causal'])
            hasCausalIn2 = any(set(seed).issubset(stem) for stem in stems2 for seed in self.seedSet['causal'])
            hasNonCausalIn1 = any(set(seed).issubset(stem) for stem in stems1 for seed in self.seedSet['notcausal'])
            hasNonCausalIn2 = any(set(seed).issubset(stem) for stem in stems2 for seed in self.seedSet['notcausal'])
            #print(stems1, stems2, hasCausalIn1, hasCausalIn2)
            if hasCausalIn1:
                if not hasNonCausalIn1:
                    yield 'causal',stems1,stems2
            else:
                if hasNonCausalIn1:
                    yield 'notcausal',stems1,stems2
            if hasCausalIn2:
                if not hasNonCausalIn2:
                    yield 'causal',stems2,stems1
            else:
                if hasNonCausalIn2:
                    yield 'notcausal',stems2,stems1

            #ignore everything else for now
            if not (hasCausalIn1 or hasCausalIn2 or hasNonCausalIn1 or hasNonCausalIn2):
                yield 'other',stems1,stems2
                yield 'other',stems2,stems1

            #if not hasCausalIn1 and not hasCausalIn2:
            #    yield 'notcausal',stems1,stems2
            #    yield 'notcausal',stems2,stems1
            
def main(iterator, withNoS1=True, withS1=True, prefix=''):
    counts = {'all':
              {'in_s1': collections.defaultdict(int),
               'in_s2': collections.defaultdict(int),
               'in_s2_not_in_s1': collections.defaultdict(int),
               'total': 0}
              }

    for filename,stems1,stems2,weighting in iterator:
        if filename not in counts:
            counts[filename] = {'in_s1': collections.defaultdict(int),
                                'in_s2': collections.defaultdict(int),
                                'in_s2_not_in_s1': collections.defaultdict(int),
                                'total': 0}
                         
        #count all the words that appear on the right side that don't appear on the left
        for stem in stems2-stems1:
            counts[filename]['in_s2_not_in_s1'][stem] += weighting(filename, stem)
            counts[filename]['in_s1'][stem] += (1-weighting(filename, stem))
            counts['all']['in_s2_not_in_s1'][stem] += weighting(filename, stem) ##correct weighting?
            counts['all']['in_s1'][stem] += (1-weighting(filename, stem))
            
        #count the total number of times each word appears on the left side and subtract from the total documents later
        for stem in stems1:
            counts[filename]['in_s1'][stem] += 1
            counts['all']['in_s1'][stem] += 1
        counts[filename]['total'] += 1
        counts['all']['total'] += 1

        #finally, count just the right-hand side
        for stem in stems2:
            counts[filename]['in_s2'][stem] += 1
            counts['all']['in_s2'][stem] += 1

    ret = {}
    for filename in counts:
        print('connective: ', filename, 'counts: ', counts[filename]['total'])
        if filename == 'all':
            continue

        #saveFilename = '{}.kld.{}.{}.{}.{}.{}.{}'.format(prefix, filename, split_sent, word_pairs, max_ngrams, location, 'not_in_s1')
        saveFilename = '{}.{}.kld.not_in_s1'.format(prefix, filename)
        
        if withNoS1:
            qp = calcNotInS1(counts[filename], counts['all'])
            kldt = KldTransformer()
            kldt.fit(*qp)
            kldt.save(saveFilename)

        if withS1:
            saveFilename = saveFilename.replace('not_in_s1', 'in_s1')
            qp = calcInS1(counts[filename], counts['all'])
            kldt = KldTransformer()
            kldt.fit(*qp)
            kldt.save(saveFilename)

        ret[filename] = (saveFilename, kldt)

    return ret

def mainstar(settings):
    return main(*settings)

def getCausalPhrases(phrases, seedSet, stem=True):
    #build the table to look up the phrases in the pairs that have causal relations in them
    causalPhrases = {'causal': set(),
                     'notcausal': set()}
    causalPhrases = {i:set() for i in seedSet.keys()}
    
    if stem:
        stemmedSeedSet = {i:[set(map(stemmer.stem, j)) for j in seedSet[i]] for i in seedSet}
    else:
        stemmedSeedSet = {i:list(map(set, seedSet[i])) for i in seedSet}
        
    print(stemmedSeedSet)
    for count,phrase in enumerate(phrases):
        if count % 100000 == 0:
            print(count)
        #if any of the causal markers appear as part of this phrase, consider it causal and add all its paraphrases
        for i in causalPhrases.keys():
            if any(stemmedSeed.issubset(phrase.split()) for stemmedSeed in stemmedSeedSet[i]):
                for paraphrase in phrases[phrase]:
                    causalPhrases[i].add(tuple(paraphrase.split()))

    #print(len(causalPhrases['causal']),
    #      len(causalPhrases['notcausal']))
    for i in causalPhrases.keys():
        print(len(causalPhrases[i]))
    return causalPhrases

def trainAndClassifyPhrases(iterator, kldt, verbose=False):
    kldf = TfkldFactorizer(kldt['causal'][0])
    classifier = SGDClassifier()
    
    #first train the data on the known points
    if verbose:
        print('getting training...')
    X = []
    labels = []
    for datapoints in iterator:
        if datapoints[0] == 'other':
            continue
        elif datapoints[0] == 'causal':
            label = 1
        else:
            label = 0
        for datapoint in datapoints[1:]:
            x = {}
            for ngram in datapoint:
                x[ngram] = 1
            X.append(x)
            labels.append(label)

    if verbose:
        print('SVD transform...')
    X = kldf.fit_transform(X)
    if verbose:
        print(X.shape)
        print(len(labels), sum(labels))
        print('training...')

    training = zip(X, labels)
    training = balance(training)
    X, labels = zip(*training)
    classifier.fit(X, labels)
    
    #TODO: should probably tune here on held out data
    
    #then classify the unknown points (leverage paraphrases? only if they both agree on classification?)
    X_unk = [[], []]
    for datapoints in iterator:
        if datapoints[0] != 'other':
            continue
        x = {}
        for ngram in datapoints[1]:
            x[ngram] = 1
        if len(X_unk[0]) == len(X_unk[1]):
            X_unk[0].append(x)
        else:
            X_unk[1].append(x)
    X_unk = [kldf.transform(i) for i in X_unk]
    print(X_unk[0].shape, X_unk[1].shape)
    labels_unk = [classifier.predict(i) for i in X_unk]
    labels_causal = labels_unk[0] & labels_unk[1]
    labels_notcausal = (1^labels_unk[0]) & (1^labels_unk[1])

    #then for each phrase, count how often it occurs as causal or not causal
    count = 0
    classifiedPhrases = {'causal': collections.defaultdict(int),
                         'notcausal': collections.defaultdict(int),
                         'total': collections.defaultdict(int)}
    for datapoints in iterator:
        if datapoints[0] != 'other':
            continue
        for ngram in datapoints[1]:
            classifiedPhrases['total'][ngram] += 1
            if labels_causal[count//2] == 1:
                classifiedPhrases['causal'][ngram] += 1
            if labels_notcausal[count//2] == 1:
                classifiedPhrases['notcausal'][ngram] += 1
                
        count += 1
    return classifiedPhrases

def addNewPhrases(data, kldt, seedSet, causalPhrases, classifiedPhrases, order, n=1, minProb=.5, minKLD=.001):

    for phraseType in order:
        topKLD = kldt[phraseType][1].topKLD()
        count = 0
        print(phraseType, len(topKLD))
        for index,kld in enumerate(topKLD):

            if count >= n or kld[3] < minKLD:
                break
            
            s = set(kld[0])
            hasMarker = any(set(marker).issubset(kld[0]) for marker in all_markers)
            alreadyFound = any(kld[0] in seedSet[i] for i in order)
            noInfoGain = any(s.issubset(seed) or set(seed).issubset(s) for seed in seedSet[phraseType])
            isInPhraseType = kld[0] in causalPhrases[phraseType]
            #kld[0][:(len(kld[0])+siblingIndex)/2] in causalPhrases[phraseType]
            isInOtherPhraseType = any(kld[0] in causalPhrases[i] for i in set(order) - {phraseType})
            #any(kld[0][:(len(kld[0])+siblingIndex)/2] in causalPhrases[i] for i in set(order) - {phraseType})
            print(count, n, index, kld, hasMarker, alreadyFound, noInfoGain, isInPhraseType, isInOtherPhraseType)
            if not hasMarker and not alreadyFound and kld[1] > kld[2] and not kld[0][0].startswith('SIBLING'):
                
                if True:
                    siblingIndex = 0
                elif kld[0][-1].startswith('SIBLING'):
                    siblingIndex = -1
                    minLength = 3
                else:
                    siblingIndex = -2
                    minLength = 4
                #len(s) == minLength and
                num = classifiedPhrases[phraseType].get(kld[0], 0)
                denom = classifiedPhrases['total'].get(kld[0], classifiedPhrases[phraseType].get(kld[0], 1))
                prob = 1.*num / denom
                print(prob, num, denom)
                #if (s & modal_auxiliary) or kld[0][siblingIndex][8] not in {'S', 'N', 'V'} or not set(kld[0]) & {'N', 'V', 'R', 'J'} or
                if noInfoGain or not isInPhraseType  or isInOtherPhraseType or prob <= minProb:
                    continue

                new_kld = list(kld[0])
                #new_kld[siblingIndex] = new_kld[siblingIndex][:9]
                seedSet[phraseType].add(tuple(new_kld)) #tuple(filter(lambda x:x.islower(), kld[0])))
                count += 1
        print(seedSet)


if __name__ == '__main__':
    if False:
        phrases,maxLen = readPhrasePairs(sys.argv[1],
                                         False)
        phrases,maxLen = readPhrasePairs(sys.argv[2],
                                         False,
                                         phrases=phrases,
                                         outfile=sys.argv[3],
                                         maxLen=maxLen)
    elif True:
        '''
        with gzip.open(sys.argv[1]) as f:
            phrases = json.load(f) 
        data = getDataNgrams(ParsedPairIterator(sys.argv[2], True),
                             phrases["phrases"],
                             phrases["maxLen"],
                             "\t") #" , ")#
        with gzip.open(sys.argv[3], "w") as f:
            json.dump(data, f)
        '''
        data = getDataNgrams2(ParsedPairIterator(sys.argv[1], True),
                             7)
        with gzip.open(sys.argv[2], "w") as f:
            json.dump(data, f)
    else:
        print('loading data...')
        with gzip.open(sys.argv[1]) as f:
            data = json.load(f)
        print('loading phrases...')
        with gzip.open(sys.argv[3]) as f:
            phrases = json.load(f) 
        
        k = 10
        n = 10
        seedSet = {'causal': wordUtils.causal_markers,
                   'notcausal': wordUtils.noncausal_markers}
        print(seedSet)
        for i in range(k):
            print(i)
            iterator = list(PreprocessedIterator(phrases=data, seedSet=seedSet))
            print('calculating kld...')
            kldt = main(iterator, withS1=False, prefix=sys.argv[2] + str(i))
            print('train and classify...')
            classifiedPhrases = {'causal': {}, 'notcausal': {}, 'total': {}} #trainAndClassifyPhrases(iterator, kldt, True)
            causalPhrases = getCausalPhrases(phrases['phrases'], seedSet, False)
            addNewPhrases(data, kldt, seedSet, causalPhrases, classifiedPhrases, ('notcausal', 'causal'), n=n, minProb=-1, minKLD=0)
            
    exit()

    import argparse
    parser = argparse.ArgumentParser(description='calculate KL divergence for a set of sentences')
    parser.add_argument('indir', 
                        help='the directory contained aligned sentences')
    parser.add_argument('--splitSent', action='store_true')
    parser.add_argument('--wordPairs', action='store_true')
    parser.add_argument('--maxNgrams', type=int, default=2)
    parser.add_argument('--location', type=int, default=0)    
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        settings = list(itertools.product([args.indir],
                                          [True, False],
                                          [True,False],
                                          list(range(1,args.maxNgrams)),
                                          [True],
                                          [True],
                                          [args.location]
                                          ))
    else:
        settings = [[args.indir,
                     args.splitSent,
                     args.wordPairs,
                     args.maxNgrams,
                     True,
                     True,
                     args.location]]
    pool = multiprocessing.Pool(processes=len(settings))
    pool.map(mainstar, settings)


