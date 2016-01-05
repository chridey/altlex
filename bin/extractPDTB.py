import sys
import argparse
import os
import re
import json

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import sent_tokenize,word_tokenize

from chnlp.utils.treeUtils import treesFromString
from chnlp.utils import wordUtils

#also POS

rangeRe = re.compile('\.\.|;')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract data from PDTB')

    parser.add_argument('indir', 
                        help='the directory containing the PDTB and WSJ ')
    parser.add_argument('outfile', 
                        help='the file to output the data to in JSON format')
    parser.add_argument('--dev', default='0,1')
    parser.add_argument('--train', default='2,22')
    parser.add_argument('--test', default='23,24')
    parser.add_argument('--explicit', action='store_true')
    args = parser.parse_args()

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    dataset = {}
    for r in (args.train,args.dev,args.test):
        dataset[r] = []
        start,end = r.split(',')
        for section in range(int(start), int(end)+1):
            currDir = os.path.join(os.path.join(args.indir,
                                                'pdtb_pipe'),
                                   '%02d') % section
            wsjDir = os.path.join(os.path.join(args.indir,
                                                'wsj'),
                                   '%02d') % section
            ptbDir = os.path.join(os.path.join(args.indir,
                                               'ptb'),
                                  '%02d') % section
            print(currDir, wsjDir)
            files = os.listdir(wsjDir)            
            for f in files:
                print(f)
                with open(os.path.join(wsjDir, f)) as wsjfp:
                    wsj = wsjfp.read()
                if os.path.exists(os.path.join(currDir, f) + '.pipe'):
                    #read in PTB trees
                    with open(os.path.join(ptbDir, f) + '.prd') as ptbfp:
                        treeString = ptbfp.read()
                    trees = treesFromString(treeString)
                    with open(os.path.join(currDir, f) + '.pipe') as pdtbfp:
                        for index,line in enumerate(pdtbfp):
                            data = line.split('|')

                            relType = data[0] #Explicit/Implicit/AltLex/EntRel/NoRel
                            if relType == 'Explicit' and not args.explicit:
                                continue

                            classes = data[11:15]
                            attrSpan = data[19]
                            spans1 = data[22],data[29],data[42]
                            spans2 = data[32],data[39],data[45]

                            '''
                            spans = spans1 + spans2 + [attrSpan]
                            start = min(int(min(rangeRe.split(x), key=int)) for x in spans if x)
                            end = max(int(max(rangeRe.split(x), key=int)) for x in spans if x)
                            sentences = wsj[start:end+1].split('\n')
                            if len(sentences) != 2:
                                print('Problems with: {}'.format(sentences))
                                continue
                            startIndex = start
                            '''

                            starts = []
                            ends = []
                            for spans in (spans1, spans2):
                                start = min(int(min(rangeRe.split(x), key=int)) for x in spans if x)
                                starts.append(start)
                                end = max(int(max(rangeRe.split(x), key=int)) for x in spans if x)
                                ends.append(end)

                            sentences = []
                            if attrSpan:
                                attrStart = int(min(rangeRe.split(attrSpan), key=int))
                                attrEnd = int(max(rangeRe.split(attrSpan), key=int))
                                if attrStart < starts[0]:
                                    starts[0] = attrStart
                                elif attrStart < starts[1]:
                                    attr = wsj[attrStart:attrEnd+1]
                                    if '.' in attr or '!' in attr or '?' in attr:
                                        ends[0] = attrEnd
                                    else:
                                        starts[1] = attrStart
                                if attrEnd > ends[1]:
                                    ends[1] = attrEnd

                            startIndex = min(starts)
                            sentences = [wsj[starts[1]:ends[1]+1],
                                         wsj[starts[0]:ends[0]+1]]

                            #if attrSpan:
                                #print(sentences)
                            sentenceInfo = [None, None]

                            #TODO: this wont match up
                            pos = zip(*trees[index].pos())
                            print(pos)
                            pos = pos[1]
                            start = 0
                            for i,s in enumerate(sentences):
                                words = word_tokenize(s)
                                sentenceInfo[i] = {'words': words}
                                sentenceInfo[i]['pos'] = pos[start:start+len(words)]
                                print(words, sentenceInfo[i]['pos'])
                                sentenceInfo[i]['lemmas'] = wordUtils.lemmatize(words, sentenceInfo[i]['pos'])
                                sentenceInfo[i]['stems'] = [stemmer.stem(j.decode('latin-1')).lower() for j in words]
                                start += len(words)
                            assert(len(pos) == start)

                            if any('Contingency' in i for i in classes):
                                tag = 'causal'
                            else:
                                tag = 'notcausal'

                            #data[3] = connective span and is always contained within a sentence
                            if data[3]:
                                connectiveStart = int(min(rangeRe.split(data[3]), key=int))
                                connectiveEnd = int(max(rangeRe.split(data[3]), key=int))
                            else:
                                connectiveEnd = None
                                connectiveStart = None
                                
                            #print(relType, classes)
                            datapoint = {'sentences': sentenceInfo,
                                         'tag': tag,
                                         'relation': relType,
                                         'classes': list(filter(None, classes)),
                                         #'connectiveStart': connectiveStart,
                                         #'connectiveEnd': connectiveEnd,
                                         'altlexLength': len(word_tokenize(wsj[connectiveStart:connectiveEnd+1])) if relType == 'AltLex' else 0}
                            #print(datapoint)
                            dataset[r].append(datapoint)

    for r in dataset:
        with open('{}.{}'.format(args.outfile, r), 'w') as f:
            json.dump(dataset[r], f, encoding='latin-1')
