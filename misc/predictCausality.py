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

def predictCausality(sentenceIterator, altlexHandler):
    for sentence in sentenceIterator:
        dataPoints = altlexHandler.dataPoints(sentence)
        labels = altlexHandler.predict(dataPoints, False)

        ranges = []
        for dataPoint, label in zip(dataPoints, labels):
            if label:
                start = len(dataPoint.getPrevWords())
                end = start + dataPoint.altlexLength
                ranges.append((start,end))
        
        yield sentence['words'],ranges
        
if __name__ == '__main__':
    data_filename = sys.argv[1]

    #a sentence iterator returns parsed sentences
    sentenceIterator = PlaintextIterator(data_filename)
    altlexHandler = AltlexHandler()
    
    labeled_sentences = predictCausality(sentenceIterator, altlexHandler)

    for sentence,ranges in labeled_sentences:
        output = ''
        last = 0
        for start,end in ranges:
            words = sentence[last:start]
            output += ' '.join(words).encode('utf-8')
            output += ' ##'
            words = sentence[start:end]
            output += ' '.join(words).encode('utf-8')
            output += '## '
            last = end
        output += ' '.join(sentence[last:]).encode('utf-8')
        
        print(output)
