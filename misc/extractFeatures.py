#iterate through the parsed pairs data and aligned altlexes
#for only the specified labels
#extract altlex
#extract dependencies
#extract features

import sys
import json

from altlex.utils.readers.alignedParsedPairIterator import AlignedParsedPairIterator 
from altlex.featureExtraction.featureExtractor import FeatureExtractor
from altlex.featureExtraction.dataPointMetadata import DataPointMetadata,DataPointMetadataList

def main(alignedLabelsIterator,
         featureExtractor,
         sentenceIndices=None,
         datumIndices=None):

    dataset = DataPointMetadataList()
    for sentenceId,datumId,dataPoint,label in alignedLabelsIterator.iterData(sentenceIndices,
                                                                             datumIndices,
                                                                             verbose=True,
                                                                             modBy=100):    
        features = featureExtractor.addFeatures(dataPoint)
        dataset.append(DataPointMetadata(dataPoint, features, label, datumId, sentenceId))
            
    return dataset
        
if __name__ == '__main__':
    parallelParseDir = sys.argv[1]
    print('loading alignments...')
    with open(sys.argv[2]) as f:
        alignments = f.read().splitlines()
    labelsFile = sys.argv[3]
    dataFile = sys.argv[4]

    if len(sys.argv) > 5:
        configFile = sys.argv[5]
        with open(configFile) as f:
            settings = json.load(f)
        featureExtractor = FeatureExtractor(settings, verbose=True)
    else:
        featureExtractor = FeatureExtractor(verbose=True)
        
    alignedLabelsIterator = AlignedParsedPairIterator(parallelParseDir,
                                                      alignments,
                                                      verbose=True)
        
    alignedLabelsIterator.load(labelsFile)

    sentenceIndices = set(alignedLabelsIterator.getValidSentenceIndices())
    print(len(sentenceIndices))
    
    #get only labeled points
    datumIndices = alignedLabelsIterator.getIndices(sentenceIndices, validate=True)
    validIndices = set()
    for label in datumIndices:
        validIndices.update(datumIndices[label])
    print(len(validIndices))
    
    #now iterate through the data, getting only those points that have labels
    #extract the altlex, dependencies, and features
    dataset = main(alignedLabelsIterator,
                   featureExtractor,
                   sentenceIndices,
                   validIndices)

    dataset.save(dataFile)
    
