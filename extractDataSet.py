#first extract sentences and their scores
#then get the rankings for the altlexes
#then determine how many of the altlexes are causal

from sentenceReader import SentenceRelationReader,extractParse,extractAltlex
from xmlReader import XMLReader

import sys
import json

def writeDataJSON(data, handle):
    fullOutput = []
    for d in data:
        sentence, prevSentence, altlex = d
        output = []
        for s in (sentence,prevSentence):
            output.append({"words": s.words,
                           "lemmas": s.lemmas,
                           "stems": s.stems,
                           "pos": s.pos,
                           "ner": s.ner,
                           "parse": s.parseString})
            
        fullOutput.append({"sentences": output,
                           "altlexLength": len(altlex),
                           "tag":sentence.tag})

    return json.dump(fullOutput, handle)

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
        if len(sys.argv) > 2:
            s = ' '.join(sentence.words)
            mx = 0
            argmax = None

            for a in altlexes:
                if s.startswith(a):
                    alist = a.split()
                    if len(alist) > mx:
                        mx = len(alist)
                        argmax = alist

            if argmax is not None:
                siblings = extractParse(argmax, sentence.parse)
                if siblings is None:
                    print("Error with altlex: {} and parse: {}".format(argmax, sentence.parseString), file=sys.stderr)
                    continue
                print(s, sentence.tag, [s.label() for s in siblings])
        elif prevSentence:
            a = extractAltlex(sentence.parse).split()
            if len(a):
                data.append((sentence, prevSentence, a))
                #s = ' '.join(sentence.words)
                #print("{}\t{}\t{}".format(s, a, sentence.tag))

        prevSentence = sentence

writeDataJSON(data, sys.stdout)
