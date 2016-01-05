import sys
import argparse
import json
import time

from chnlp.utils.readers.parsedGigawordReader import ParsedGigawordReader
from chnlp.utils.readers.fastParsedGigawordReader import FastParsedGigawordReader
from chnlp.utils.readers.sentenceReader import SentenceReader
from chnlp.utils.readers.xmlReader import XMLReader
from chnlp.utils.readers.distantSupervisionReader import DistantSupervisionReader
from chnlp.utils.treeUtils import extractAltlex

from chnlp.altlex.readAltlexes import readAltlexes, matchAltlexes, distantMatchAltlexes
from chnlp.altlex.readers.sentenceRelationReader import SentenceRelationReader
from chnlp.altlex.writeDataJSON import writeDataJSON

parser = argparse.ArgumentParser(description='Create a data set of altlexes.')

parser.add_argument('infile', 
                    help='the file or directory containing the sentences and metadata')
parser.add_argument('--outfile', metavar='O', type=argparse.FileType('w'),
                    help='the name of the file to write JSON output to (default: stdout)')

parser.add_argument('-a', '--altlexes', metavar='A',
                    help='the name of the file containing the altlexes, optional')

parser.add_argument('--distantAltlexes', action='store_true',
                    help='dont do exact matches for altlexes')

parser.add_argument('-d', '--distant', action='store_true',
                    help='extract data for distant supervision of causality')

parser.add_argument('--all', action='store_true',
                    help='use all data whether it contains an altlex or not')

parser.add_argument('-u', '--unsupervised', action='store_true',
                    help='flag to indicate data is not tagged')

parser.add_argument('--xml', action='store_true',
                    help='flag to indicate XML format (default: JSON)')

parser.add_argument('--gz', action='store_true',
                    help='flag to indicate gzipped data (default: not compressed)')

parser.add_argument('-t', '--timeout', metavar='T', type=float, default=float('inf'),
                    help='timeout after T seconds instead of reading entire file')

parser.add_argument('-n', '--maxPoints', metavar='N', type=float, default=float('inf'),
                    help='stop after collecting N datapoints')

parser.add_argument('--logPoints', metavar='L', type=float, default=float('inf'),
                    help='write to outfile every L datapoints')

args = parser.parse_args()

if args.outfile:
    outfile = args.outfile
else:
    outfile = sys.stdout

if args.altlexes:
    altlexes = readAltlexes(args.altlexes)

if args.xml and args.gz:
    if args.distant:
        reader = FastParsedGigawordReader
    else:
        reader = ParsedGigawordReader
elif args.xml:
    reader = XMLReader
else:
    raise NotImplementedError

if args.distant:
    sentenceReader = DistantSupervisionReader
elif args.unsupervised:
    sentenceReader = SentenceReader
else:
    sentenceReader = SentenceRelationReader

data = []
starttime = time.time()

try:
    r = reader(args.infile)

    for s in r.iterFiles():
        sr = sentenceReader(s)
        prevSentence = None

        for sentence in sr.iterSentences():
            if args.distant:
                #print(sentence.current.words, sentence.previous.words, sentence.current.parse,
                #      sentence.current.dependencies)
                data.append((sentence.current, sentence.previous, []))
            else:
                match = None

                if prevSentence is not None:
                    if args.altlexes:
                        if args.distantAltlexes:
                            match = distantMatchAltlexes(sentence, altlexes)
                        else:
                            match = matchAltlexes(sentence, altlexes)
                    elif not args.all:
                        match = extractAltlex(sentence.parse).split()

                    if match:
                        #print(prevSentence.words)
                        #print('sentence:', match)
                        #print(sentence.tag)
                        data.append((sentence, prevSentence, match))
                    elif args.all:
                        data.append((sentence, prevSentence, []))
                
            prevSentence = sentence

            if len(data) % args.logPoints == 0:
                print(len(data))
                writeDataJSON(data, outfile)
                
            if time.time() - starttime > args.timeout or len(data) > args.maxPoints:
                raise Exception
            
except KeyboardInterrupt:
    print ("Terminating on keyboard interrupt")
except Exception as e:
    writeDataJSON(data, outfile)
    raise e

writeDataJSON(data, outfile)

