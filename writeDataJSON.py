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

        tag = getattr(sentence, 'tag', None)
            
        fullOutput.append({"sentences": output,
                           "altlexLength": len(altlex),
                           "tag":tag})

    return json.dump(fullOutput, handle)
