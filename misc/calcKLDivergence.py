import sys
import collections

from altlex.utils.readers.alignedParsedPairIterator import AlignedParsedPairIterator 
from altlex.ml.tfkld import KldTransformer, TfkldFactorizer

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

def main(iterator, withS1=False, verbose=False):
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
        if verbose:
            print('connective: ', filename, 'counts: ', counts[filename]['total'])
        if filename == 'all':
            continue

        if withS1:
            qp = calcNotInS1(counts[filename], counts['all'])
        else:
            qp = calcInS1(counts[filename], counts['all'])

        kldt = KldTransformer()
        kldt.fit(*qp)

        ret[filename] = kldt

    return ret


if __name__ == '__main__':
    parallelParseDir = sys.argv[1]
    print('loading alignments...')
    with open(sys.argv[2]) as f:
        alignments = f.read().splitlines()
    labelsFile = sys.argv[3]
    suffix = sys.argv[4]
    combined = int(sys.argv[5])
    
    if len(sys.argv) > 6:
        percentTrain = float(sys.argv[6])
    
    alignedLabelsIterator = AlignedParsedPairIterator(parallelParseDir,
                                                      alignments,
                                                      combined=combined)
    alignedLabelsIterator.load(labelsFile)

    if len(sys.argv) > 6:
        maxIndex = int(percentTrain*alignedLabelsIterator.numSentences)
        sentenceIndices = set(range(maxIndex))
    else:
        sentenceIndices = None
        
    kldt = main(alignedLabelsIterator.iterLabeledAltlexPairs(sentenceIndices=sentenceIndices),
                withS1=False,
                verbose=True)
    
    for kldt_type in kldt:
        kldt[kldt_type].save(kldt_type + suffix)
