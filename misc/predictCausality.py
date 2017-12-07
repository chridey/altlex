#read data (may be in crowdflower format)
#tokenize words and sentences, parse, POS-tag, etc
#split sentences on altlex using known causal altlexes
#modify data points as needed (splitDependencies)
#add features
#make interaction features
#make predictions using default, bootstrapped model

import sys

from altlex.utils.readers.plaintextIterator import PlaintextIterator

from altlex.featureExtraction.altlexHandler import AltlexHandler

if __name__ == '__main__':
    data_filename = sys.argv[1]

    #a sentence iterator returns parsed sentences
    sentenceIterator = PlaintextIterator(data_filename)
    altlexHandler = AltlexHandler()

    sentences = list(sentenceIterator)
    altlex_locations = altlexHandler.findAltlexes(sentences)

    output = [i['words'] for i in sentences]
    for label, index, start, end in altlex_locations:
        if label:
            output[index][start] = '##' + output[index][start]
            output[index][end] += '##'
            
    print(' '.join(output).encode('utf-8'))
    
