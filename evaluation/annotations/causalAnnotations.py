from __future__ import print_function
import sys
import os
import re
import json
import gzip

from altlex.misc import extractFeatures
from altlex.featureExtraction.featureExtractor import FeatureExtractor

from altlex.utils import wordUtils
from altlex.utils.readers.annotatedParsedIterator import AnnotatedParsedIterator

altlexRegex = re.compile('##.+?##')
replaceRegex = re.compile('((--)|(\+\+)) ?((reason)|(result))')

def getAnnotatedFileData(f):
    ret = {'sentences': {}}
    for line in f:
        if line.startswith('Title: '):
            ret['title'] = line.replace('Title: ', '').strip()
        elif ':' in line and line[0].isdigit():
            sentenceIndex, sentence = line.strip().split(':', 1)
            ret['sentences'][int(sentenceIndex)] = sentence
    return ret

def extractAltlexes(indir):
    markedData = {}
    totalAltlexes = 0

    #first go through annotated files and add to lookup table
    for filename in os.listdir(indir):
        print(filename)
        if not filename.endswith('0') and not filename.endswith('1'):
            continue

        articleIndex, wikiIndex = filename.split('.')
        with open(os.path.join(indir, filename)) as f:
            data = getAnnotatedFileData(f)

        sentences = []
        for index in data['sentences']:
            line = data['sentences'][index]
            altlexes = []
            for altlex in altlexRegex.findall(line):
                print(altlex)
                if '--' in altlex:
                    altlex,category = altlex.replace('##', '').split('--')
                else:
                    altlex,category = altlex.replace('##', '').split('++')
                    
                altlex = altlex.split()
                category = category.strip()
                if category == 'reason':
                    classes = ('Contingency.Cause.Reason',)
                elif category in ('result', 'restul'):
                    classes = ('Contingency.Cause.Result',)
                else:
                    classes = tuple()
                    print(altlex,category)
                altlexes.append((altlex,category))

            sentence = line.replace('##', ' ').replace('{{', ' ').replace('}}', ' ')
            sentence = replaceRegex.sub(' ', sentence)
                
            sentences.append((altlexes,sentence))
            totalAltlexes += len(altlexes)
        markedData[(int(articleIndex),int(wikiIndex))] = sentences

    return markedData,totalAltlexes

def writeFormattedData(markedData, outdir):
    for articleIndex,wikiIndex in markedData:
        with open(os.path.join(outdir,
                               '{}.{}'.format(articleIndex,wikiIndex)),
                  'w') as f:

            for altlexes,sentence in markedData[(articleIndex,wikiIndex)]:
                print(sentence, file=f)

def getDataForParsing(indir, outdir):
    markedData, totalAltlexes = extractAltlexes(indir)
    print(totalAltlexes)
    writeFormattedData(markedData, outdir)

def getCausalAnnotations(indir, altlexes, featureExtractor, labelLookup, wordsOnly=False, verbose=False):
    markedData, totalAltlexes = extractAltlexes(indir)

    iterator = AnnotatedParsedIterator(indir, markedData, altlexes, labelLookup, verbose=verbose,
                                       wordsOnly=wordsOnly)

    featureset = extractFeatures.main(iterator, featureExtractor)

    if verbose:
        print('Total Found : {}'.format(iterator.totalAltlexes))
        print('Total Not Found: {}'.format(iterator.unfoundAltlexes))

    return featureset

def getTextOnly(indir, altlexes, labelLookup):
    markedData, totalAltlexes = extractAltlexes(indir)
    
    iterator = AnnotatedParsedIterator(indir, markedData, altlexes, labelLookup)

    for sentenceIndex,datumIndex,prevWords,altlex,currWords in iterator.iterData(textOnly=True):
        print("{}\t{}\t{}\t{}\t{}".format(sentenceIndex,
                                          datumIndex,
                                          prevWords.encode('utf-8'),
                                          altlex.encode('utf-8'),
                                          currWords.encode('utf-8')))

def saveDiscoveredAltlexes(indir, outfile):
    markedData, totalAltlexes = extractAltlexes(indir)

    uniqueAltlexes = set()
    for fileIndex in markedData:
        for altlexes,sentence in markedData[fileIndex]:
            for altlex,metaLabel in altlexes:
                uniqueAltlexes.add(tuple(i.lower() for i in altlex))

    with open(outfile, 'w') as f:
        for altlex in sorted(uniqueAltlexes):
            print(' '.join(altlex), file=f)
            
if __name__ == '__main__':

    indir = sys.argv[1]
    outfile = sys.argv[2]
    
    with open(sys.argv[3]) as f:
        altlexes = {tuple(i.split()) for i in f.read().splitlines()}    

    getTextOnly(indir, altlexes, wordUtils.trinaryCausalSettings[1])
    exit()

    if len(sys.argv) > 4:
        configFile = sys.argv[4]
        with open(configFile) as f:
            settings = json.load(f)
        featureExtractor = FeatureExtractor(settings, verbose=True)
    else:
        featureExtractor = FeatureExtractor(verbose=True)

    data = getCausalAnnotations(indir,
                                altlexes,
                                featureExtractor,
                                wordUtils.trinaryCausalSettings[1],
                                wordsOnly=False,
                                verbose=True)

    data.save(outfile)
