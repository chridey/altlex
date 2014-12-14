#first extract sentences and their scores
#then get the rankings for the altlexes
#then determine how many of the altlexes are causal

from sentenceReader import SentenceRelationReader,extractParse,extractAltlex
from xmlReader import XMLReader
from writeDataJSON import writeDataJSON

import sys
import json

xr = XMLReader(sys.argv[1])
if len(sys.argv) > 2:
    with open(sys.argv[2]) as f:
        altlexes = f.read().splitlines()

    for i in range(len(altlexes)):
        altlexes[i] = altlexes[i].replace("'s", " 's")
        altlexes[i] = altlexes[i].replace(":", " :")
        altlexes[i] = altlexes[i].replace(",", " ,")
        altlexes[i] = altlexes[i].replace('"', ' "')

#need to repeat this for the marked altlexes but without expanding them

data = []
for f in xr.iterFiles():
    #print(f)
    
    srr = SentenceRelationReader(f)
    prevSentence = None

    for sentence in srr.iterSentences():
        if prevSentence:
            if len(sys.argv) > 2:
                #s = ' '.join(sentence.words)
                s = sentence.words
                mx = 0
                argmax = None

                for a in altlexes:
                    #if s.startswith(a):
                    alist = a.split()
                    if s[:len(alist)] == alist:
                        if len(alist) > mx:
                            mx = len(alist)
                            argmax = alist

                if argmax is not None:
                    print('found {} in {}: {}'.format(argmax, sentence.words, sentence.tag), file=sys.stderr)
                    #siblings = extractParse(argmax, sentence.parse)
                    #if siblings is None:
                    #    print("Error with altlex: {} and parse: {}".format(argmax, sentence.parseString), file=sys.stderr)
                    #    continue
                    data.append((sentence, prevSentence, argmax))
                    #print(s, sentence.tag, [s.label() for s in siblings])
            else:
                a = extractAltlex(sentence.parse).split()
                if len(a):
                    data.append((sentence, prevSentence, a))
                    #s = ' '.join(sentence.words)
                    #print("{}\t{}\t{}".format(s, a, sentence.tag))

        prevSentence = sentence

writeDataJSON(data, sys.stdout)
