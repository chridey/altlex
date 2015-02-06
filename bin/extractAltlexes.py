import sys
import argparse
import json
import time

from chnlp.utils.readers.parsedGigawordReader import ParsedGigawordReader
from chnlp.utils.readers.sentenceReader import SentenceReader
from chnlp.utils.readers.xmlReader import XMLReader
from chnlp.utils.treeUtils import extractAltlex

from chnlp.altlex.readAltlexes import readAltlexes, matchAltlexes
from chnlp.altlex.readers.sentenceRelationReader import SentenceRelationReader
from chnlp.altlex.writeDataJSON import writeDataJSON

parser = argparse.ArgumentParser(description='Create a data set of altlexes.')

parser.add_argument('infile', 
                    help='the file or directory containing the sentences and metadata')
parser.add_argument('--outfile', metavar='O', type=open,
                    help='the name of the file to write JSON output to (default: stdout)')

parser.add_argument('-a', '--altlexes', metavar='A',
                    help='the name of the file containing the altlexes, optional')

parser.add_argument('--all', action='store_true',
                    help='use all data whether it contains an altlex or not')

parser.add_argument('-u', '--unsupervised', action='store_true',
                    help='flag to indicate data is not tagged')

parser.add_argument('--xml', action='store_true',
                    help='flag to indicate XML format (default: JSON)')

parser.add_argument('--gz', action='store_true',
                    help='flag to indicate gzipped data (default: not compressed)')

parser.add_argument('-t', '--timeout', metavar='T', type=float,
                    help='timeout after T seconds instead of reading entire file')

parser.add_argument('-n', '--max', metavar='N', type=float,
                    help='stop after collecting N datapoints')

args = parser.parse_args()

if args.outfile:
    outfile = args.outfile
else:
    outfile = sys.stdout

if args.timeout:
    timeout = args.timeout
else:
    timeout = float('infinity')

if args.max:
    maxPoints = args.max
else:
    maxPoints = float('infinity')

if args.altlexes:
    altlexes = readAltlexes(args.altlexes)

if args.xml and args.gz:
    reader = ParsedGigawordReader
elif args.xml:
    reader = XMLReader
else:
    raise NotImplementedError

if args.unsupervised:
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
            match = None

            if prevSentence is not None:
                if args.altlexes:
                    match = matchAltlexes(sentence.words, altlexes)
                else:
                    match = extractAltlex(sentence.parse).split()

                if match:
                    data.append((sentence, prevSentence, match))
                elif args.all:
                    data.append((sentence, prevSentence, []))
                
            prevSentence = sentence

            if time.time() - starttime > timeout or len(data) > maxPoints:
                raise Exception

except Exception:
    pass

writeDataJSON(data, outfile)

