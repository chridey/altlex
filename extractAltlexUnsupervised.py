import sys
import time

from parsedGigawordReader import ParsedGigawordReader
from sentenceReader import SentenceReader,extractAltlex
from writeDataJSON import writeDataJSON

if __name__ == '__main__':
    pgr = ParsedGigawordReader(sys.argv[1])
    timeout = int(sys.argv[2])
    
    altlexFile = open('newaltlexes', 'w')
    output = []
    starttime = time.time()

    try:
        for s in pgr.iterFiles():
                        
            sr = SentenceReader(s)
            prevSentence = None

            for sentence in sr.iterSentences():
                a = extractAltlex(sentence.parse).split()
                if len(a) and prevSentence is not None:
                    output.append((sentence, prevSentence, a))

                prevSentence = sentence

            if time.time() - starttime > timeout:
                raise Exception

    finally:
        writeDataJSON(output, altlexFile)
        altlexFile.close()

