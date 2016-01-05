import random
import argparse
import time

import numpy

import gensim

from chnlp.utils.readers.fastParsedGigawordReader import FastParsedGigawordReader
from chnlp.utils.readers.distantSupervisionReader import DistantSupervisionReader, DistantSupervisionDiscourseReader

from chnlp.nn.causalRNN import model as causalModel
from chnlp.nn.discourseRNN import model as discourseModel

parser = argparse.ArgumentParser(description='Extract causa/noncausal pairs and train with SGD.')

parser.add_argument('infile', 
                    help='the file or directory containing the sentences and metadata')
parser.add_argument('outfile', 
                    help='the name of the file to save the model to')
parser.add_argument('--numTesting', type=int, default=5000,
                    help='number of data points to set aside for testing')
parser.add_argument('-t', '--timeout', metavar='T', type=float, default=float('inf'),
                    help='timeout after T seconds instead of reading entire file')

parser.add_argument('-n', '--maxPoints', metavar='N', type=float, default=float('inf'),
                    help='stop after collecting N datapoints')

parser.add_argument('-l', '--logPoints', metavar='L', type=int, default=10000,
                    help='print results after training on L points')

parser.add_argument('--discourse', action='store_true')
parser.add_argument('--hiddenLayers', type=int, choices=(1,2), default=1)

args = parser.parse_args()

reader = FastParsedGigawordReader
initSettings = {'nh1': 100,
                'hl': args.hiddenLayers}
settings = {'seed': 31415,
            'lr': 0.05,
            'epochs': 50}

if args.discourse:
    sentenceReader = DistantSupervisionDiscourseReader
    model = discourseModel
    initSettings['nc'] = 102
else:
    sentenceReader = DistantSupervisionReader
    model = causalModel
    initSettings['nc'] = 2
    
g = gensim.models.Word2Vec.load('/local/nlp/chidey/model.1428650472.31.word2vec.curr.nelemmas')
numpy.random.seed(settings['seed'])
random.seed(settings['seed'])
initSettings['de'] = g.layer1_size
m = model(**initSettings)

starttime = time.time()
punct = {'!', '-', ',', '.', '?'}
for i in range(settings['epochs']):
    #balance these?
    causalTesting = []
    noncausalTesting = []
    training = 0

    r = reader(args.infile)
    prev_y = None
    for s in r.iterFiles():
        sr = sentenceReader(s)

        for sentence in sr.iterSentences():
            if args.discourse:
                cause = sentence.previous
                effect = sentence.current
                y = sentence.index
                #print(cause.words, effect.words, sentence.connective, y)
            else:
                if sentence.tag == 'causal':
                    cause = sentence.current
                    effect = sentence.previous
                elif sentence.tag == 'notcausal':
                    cause = sentence.previous
                    effect = sentence.current
                else:
                    print("WTF not causal or notcausal")
                    raise Exception
                y = int(sentence.tag == 'causal')
                
            x1 = [g[w.lower()] for w in cause.nelemmas if w.lower() in g and w not in punct]
            x2 = [g[w.lower()] for w in effect.nelemmas if w.lower() in g and w not in punct]
            if not len(x1) or not len(x2):
                continue

            if len(causalTesting)+len(noncausalTesting) < args.numTesting:
                if args.discourse:
                    noncausalTesting.append(((x1, x2), y))
                else:
                    if y:
                        if len(causalTesting) < args.numTesting / 2.0:
                            causalTesting.append(((x1, x2), y))
                    else:
                        if len(noncausalTesting) < args.numTesting / 2.0:
                            noncausalTesting.append(((x1, x2), y))
                    print(len(causalTesting), len(noncausalTesting))
                continue
                
            #balance?
            if not args.discourse:
                if y==prev_y:
                    continue
            prev_y=y
            m.train(x1, x2, [y], settings['lr'])
            training+=1
            if training % args.logPoints == 0:
                positives = [m.classify(*t[0]) for t in causalTesting]
                negatives = [m.classify(*t[0]) for t in noncausalTesting]
                #print(positives, negatives)
                if args.discourse:
                    #print([i[1] for i in noncausalTesting])
                    numCorrect = sum(noncausalTesting[i][1] == negatives[i] \
                                     for i in range(len(noncausalTesting)))
                else:
                    truePositives = sum(i>.5 for i in positives)
                    trueNegatives = sum(i<.5 for i in negatives)
                    numCorrect = truePositives + trueNegatives

                    print(truePositives, trueNegatives, numCorrect)
                print('Accuracy: {}'.format(1.0*numCorrect/args.numTesting))
                try:
                    os.mkdir(args.outfile)
                except Exception:
                    print('cant make dir {}'.format(args.outfile))
                m.save(args.outfile)
