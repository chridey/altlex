import sys

from altlex.utils import wordUtils
from altlex.utils.readers.alignedParsedPairIterator import AlignedParsedPairIterator

if __name__ == '__main__':
    parallelParseDir = sys.argv[1]
    print('loading alignments...')
    with open(sys.argv[2]) as f:
        alignments = f.read().splitlines()
    prefix = sys.argv[3]
    initLabelsFile = prefix + '_initLabels.json.gz'

    if sys.argv[4] == '1':
        seedSet, labelLookup = wordUtils.binaryCausalSettings
    else:
        seedSet, labelLookup = wordUtils.trinaryCausalSettings
        
    alignedLabels = AlignedParsedPairIterator(parallelParseDir, alignments)
    alignedLabels.makeLabels(seedSet, labelLookup)
    alignedLabels.save(initLabelsFile)
    
