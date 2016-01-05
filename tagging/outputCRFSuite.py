#read in all sentences in json format
#ignore the ones that start with an explicit discourse marker (actually, keep them?)
#only output the second sentence, but each word should be marked with features
#possible tags:
#B-CAL (begin causal altlex)
#I-CAL
#B-AL (begin altlex)
#I-AL
#B-CDM (causal explicit marker)
#I-CDM
#B-DM (explicit marker)
#I-DM
#O (everything else)

import json
import sys
import re

from collections import defaultdict

from chnlp.altlex.featureExtractor import makeDataset
from chnlp.altlex.config import Config
from chnlp.altlex.dataPoint import DataPoint

from chnlp.utils.utils import balance

s = re.compile('[\x80-\xff]')

def getContextTag(j, dp):
    if j >= dp.altlexLength:
        return "O"

    tag = ""
    if j==0:
        tag += "B-"
    else:
        tag += "I-"

    if dp.getTag() == 'causal':
        tag += "C"

    tag += "AL"

    return tag

config = Config()

prevWindow = 2
nextWindow = 2

with open(sys.argv[1]) as f:
    data = json.load(f)

finalData = defaultdict(list)
for i,dataPoint in enumerate(data):
    if i > float('inf'):
        break
    
    dp = DataPoint(dataPoint)

    '''
    print(dp.getCurrWords())
    print(dp.altlexLength)
    print(dp.getTag())
    '''

    if dp.altlexLength == 0:
        sentenceTag = 'O'
    elif dp.getTag() == 'notcausal':
        sentenceTag = 'AL'
    elif dp.getTag() == 'causal':
        sentenceTag = 'CAL'
    else:
        raise NotImplementedError

    tagsAndFeatures = ''
    precalculatedFeatures = config.featureExtractor.addFeatures(dp, {'marker_cosine_with_wpmodel': True})
    for j,word in enumerate(dp.getCurrWords()):
        short = dp.shorten(prevWindow, j+1, nextWindow)
        #print(short.getCurrWords())
        #features = config.featureExtractor.addFeatures(short, config.featureExtractor.taggingSettings - {'marker_cosine_with_wpmodel'})
        features = config.featureExtractor.addFeatures(short, config.featureExtractor.taggingSettings)
        features.update(precalculatedFeatures)

        tag = getContextTag(j, dp)
        print(tag, end="\t")
        tagsAndFeatures += tag + "\t"
        
        for feature in features:
            if type(features[feature]) == float:
                features[feature] = int(10*features[feature])

            fn = feature
            f = features[feature]
            if type(features[feature]) == bool:
                if features[feature] == False:
                    continue
                #need to convert categorical binary features back to categorical
                for featureName in ['altlex_marker', 'head_verb_altlex'] + \
                        ['verbnet_class_' + i for i in ('prev', 'curr', 'altlex')] + \
                        [i + 'altlex_stem: ' for i in ('uni: ', 'bi: ')] + \
                        [i + 'altlex_pos: ' for i in ('uni: ', 'bi: ')] + \
                        ['right_siblings', 'self_category', 'parent_category']:
                                    
                    if feature.startswith(featureName):
                        f = feature.replace(featureName, '')
                        fn = featureName

            try:
                print(fn + "=" + str(f), end="\t")
                #tagsAndFeatures += '{}={}\t'.format(fn, f)
            except UnicodeEncodeError:
                print(fn + "=", end="")
                #tagsAndFeatures += '{}='.format(fn)
                for i in f:
                    if ord(i) < 128:
                        print(i, end="")
                        #tagsAndFeatures += '{}'.format(i)
                print(end="\t")
                #tagsAndFeatures += "\t"
        print()
        #tagsAndFeatures += "\n"
        
    print()
    #finalData[sentenceTag].append(tagsAndFeatures)

    #TODO add mixed ngrams - I_VB_the_NNP (replace open class with POS)

    '''
    try:
        print(tagsAndFeatures)
    except UnicodeEncodeError:
        for i in tagsAndFeatures:
            if ord(i) < 128:
                print(i, end="")
        print("\n")
    '''


