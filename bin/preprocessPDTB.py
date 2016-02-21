import sys
import gzip
import json
import collections
import os

from sklearn.externals import joblib

from chnlp.misc import calcKLDivergence
from chnlp.misc import extractWikipediaAltlex

from chnlp.altlex.config import Config
from chnlp.altlex.dataPoint import DataPoint, replaceNonAscii
from chnlp.altlex import featureExtractor
from chnlp.ml.sklearner import Sklearner

from chnlp.utils import wordUtils

#from chnlp.word2vec import sentenceRepresentation

seedSet = {'causal': [set(i) for i in wordUtils.causal_markers],
           'notcausal': [set(i) for i in wordUtils.noncausal_markers]}
klds = {i:'{}.{}.kld.not_in_s1'.format(sys.argv[1], i) for i in ('causal', 'notcausal', 'other')}
#seedSet = {'reason': [set(i) for i in wordUtils.reason_markers],
#           'result': [set(i) for i in wordUtils.result_markers],
#           'notcausal': [set(i) for i in wordUtils.noncausal_markers]}
#klds = {i:'{}.{}.kld.not_in_s1'.format(sys.argv[1], i) for i in ('reason', 'result', 'notcausal', 'other')}

datasets = []
for filename in sys.argv[3:]:
    with gzip.open(filename) as f:
        dataset = json.load(f)
    outFilename = 'processed.' + filename
    exists = os.path.exists(outFilename)
    datasets.append((dataset, outFilename, exists))

'''
for dataset,filename in datasets:
    print(filename, len(dataset))
    output = []
    for datapoint in dataset:
        label = datapoint['tag'] == 'causal'
        if not datapoint['knownAltlex']:
            continue
        #if 'Other' in datapoint['classes']:
        #    label = 0
        #elif 'Contingency.Cause.Reason' in datapoint['classes']:
        #    label = 3
        #elif 'Contingency.Cause.Result' in datapoint['classes']:
        #    label = 4

        dp = DataPoint(datapoint)
        #add kld and phrase lookup features
        altlex_ngram = tuple(dp.getAltlexLemmatized())
        features = {}
        for i in altlex_ngram:
            features['altlex_stem_' + i] = 1
        for i in range(len(altlex_ngram)-1):
            features['altlex_stem_' + altlex_ngram[i] + '_' + altlex_ngram[i+1]] = 1
        output.append((features, label))

    with gzip.open(filename, 'w') as f:
        json.dump(output, f)
exit()    
'''

if not all(i[2] for i in datasets):
    #load kl divergence
    print(klds)
    deltaKLD = collections.defaultdict(dict)
    for phraseType in klds.keys():
        print(phraseType)
        kldt = joblib.load(klds[phraseType])
        topKLD = kldt.topKLD()
        for kld in topKLD:
            if kld[1] > kld[2]:
                score = kld[3]
            else:
                score = -kld[3]
            deltaKLD[phraseType][kld[0]] = score
    for q in deltaKLD:
        print(q, len(deltaKLD[q]))

    print('loading phrases...')
    with gzip.open(sys.argv[2]) as f:
        phrases = json.load(f) 

    #calculate causal phrases from starting seeds
    print('calculating causal mappings at...')
    causalPhrases = calcKLDivergence.getCausalPhrases(phrases['phrases'], seedSet, stem=False)

config = Config()

wiki = '/proj/nlp/users/chidey/parallelwikis4.json.gz'
model = '/local/nlp/chidey/model.wikipairs.doc2vec.pairwise.words'
sclient = None #sentenceRepresentation.PairedSentenceEmbeddingsClient(wiki, model)    

for dataset,filename,exists in datasets:
    print(filename, len(dataset))

    if exists:
        with gzip.open(filename) as f:
            output = json.load(f)
        assert(len(output) == len(dataset))
    else:
        output = []
    for index,datapoint in enumerate(dataset):
        dp = DataPoint(datapoint)
        if not exists:
            label = datapoint['tag'] == 'causal'
            if not datapoint['knownAltlex']:
                continue
            #if 'Other' in datapoint['classes']:
            #    label = 0
            #elif 'Contingency.Cause.Reason' in datapoint['classes']:
            #    label = 3
            #elif 'Contingency.Cause.Result' in datapoint['classes']:
            #    label = 4

            featureSettings = extractWikipediaAltlex.featureSettings
            featureSettings.update(extractWikipediaAltlex.newFeatureSettings)
            features = config.featureExtractor.addFeatures(dp, featureSettings)
            #add kld and phrase lookup features
            altlex_ngram = tuple(dp.getAltlexLemmatized())
            altlex_pos = tuple(dp.getAltlexPos())
            #TODO: make sure this is lowercase

            features['sentences'] = dp.getPrevWords() + dp.getCurrWords()
            features['altlex'] = altlex_ngram + altlex_pos
            for klass in klds:
                features[klass + '_kld'] = deltaKLD[klass].get(altlex_ngram + altlex_pos, 0)
            for klass in seedSet:
                features['in_' + klass] = altlex_ngram in causalPhrases[klass]
            features['known_altlex'] = datapoint['knownAltlex']
            features['discovered_altlex'] = datapoint['discoveredAltlex']

            #a_words = [i.lower() for i in dp.getPrevWords()]
            #b_words = [i.lower() for i in dp.getCurrWordsPostAltlex()]
            #features['a_embedding'] = sclient.infer(a_words)
            #features['b_embedding'] = sclient.infer(b_words)

            output.append((features, label))
        else:
            output[index][0]['sentences'] = dp.getPrevWords() + dp.getCurrWords()
    with gzip.open(filename, 'w') as f:
        json.dump(output, f)
    

