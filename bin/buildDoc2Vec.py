import sys
import os
import math
import json
import time
import argparse
import logging

import numpy as np

import wtmf

#import pstats, cProfile

#import pyximport
#pyximport.install()

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models.word2vec import Word2Vec
from chnlp.word2vec.sentenceIterator import Sentence

from chnlp.utils.readers.parsedGigawordReader import ParsedGigawordReader
from chnlp.utils.readers.wikipediaReader import ExtractedWikipediaReader,ParallelDataReader
from chnlp.utils.readers.sentenceReader import SentenceReader
from chnlp.utils.readers.fastParsedGigawordReader import FastParsedGigawordReader
from chnlp.utils.readers.fastSentenceReader import FastSentenceReader

from chnlp.utils.readers.gigawordReader import GigawordReader,GigawordSentenceReader,GigawordSentenceReaderLemmatized,GigawordSentenceReaderStemmed

from chnlp.word2vec.sentenceIterator import LabeledLineSentence, ParallelSentences, PreviousSentence,CurrentSentence,SentencePair,UnlabeledSentencePair,UnlabeledSentence
from chnlp.word2vec.matcher import Matcher

from chnlp.ml.tfkld import TfkldFactorizer

parser = argparse.ArgumentParser(description='train a doc2vec model on sentences in context')

parser.add_argument('infiles', nargs='+', 
                    help='the file containing the sentences and metadata in XML format')

parser.add_argument('--wordtype', '-w', metavar='W',
                    choices = {'words', 'lemmas', 'stems'},
                    default = 'words',
                    help = 'what to use as root (default: %(default)s) (choices: %(choices)s)')

parser.add_argument('-t', '--timeout', metavar='T', type=float, default=float('inf'),
                    help='timeout after T seconds instead of reading entire file')

parser.add_argument('-n', '--max', metavar='N', type=float, default=float('inf'),
                    help='stop after collecting N datapoints')

parser.add_argument('-s', '--sentence',
                    choices = {'previous', 'current', 'pairwise'},
                    default = 'current',
                    help='build model on previous or pairwise sentences rather than current, default is current')

parser.add_argument('-b', '--buildType',
                    choices = {'word2vec', 'doc2vec', 'wtmf', 'nmf'},
                    default = 'word2vec',
                    help='use doc2vec instead of word2vec, default is word2vec')

parser.add_argument('-c', '--corpus',
                    choices = {'parsed_gigaword', 'gigaword', 'extracted_wikipedia', 'parallel_wikipedia'},
                    default = 'parsed_gigaword',
                    help='use regular gigaword, default is parsed gigaword')

parser.add_argument('-m', '--matches', metavar='M',
                    help='file of sentence beginnings to filter the current sentence')

parser.add_argument('--mc', type=int, default=10,
                    help='min count of word to appear')

parser.add_argument('--n_components', type=int, default=100,
                    help='dimension of word embedding')

parser.add_argument('--max_iter', type=int, default=20,
                    help='max iterations')

parser.add_argument('--n_jobs', type=int, default=1,
                    help='number of jobs')

parser.add_argument('--weights',
                    help='term weighting for nmf, required nmf is set')

parser.add_argument('--filter',
                    help='path to file listing names of files or articles, one per line, will only do these files')

parser.add_argument('--outfile')

parser.add_argument('-v', '--verbosity', type=int, choices = {0,1,2,3}, default=0)

args = parser.parse_args()

sentenceIterator = LabeledLineSentence
        
pairs = False
if args.sentence == 'pairwise':
    pairs = True

labels = False
if args.buildType == 'doc2vec':
    labels = True
    toVec = Doc2Vec
    sentenceType = LabeledSentence
elif args.buildType == 'word2vec':
    toVec = Word2Vec
    sentenceType = Sentence
else: #if args.buildType == 'wtmf'
    toVec = None
    sentenceType = Sentence

if args.corpus == 'gigaword':
    fileReader = GigawordReader
    wordType = args.wordtype
    if wordType == 'words':
        sentenceReader = GigawordSentenceReader
    elif wordType == 'lemmas':
        sentenceReader = GigawordSentenceReaderLemmatized
    elif wordType == 'stems':
        sentenceReader = GigawordSentenceReaderStemmed
elif args.corpus == 'parsed_gigaword':
    fileReader = FastParsedGigawordReader #ParsedGigawordReader
    sentenceReader = FastSentenceReader #SentenceReader
    wordType = 'nelemmas'
elif args.corpus == 'extracted_wikipedia':
    fileReader = ExtractedWikipediaReader
    sentenceReader = GigawordSentenceReader
    wordType = 'words'
elif args.corpus == 'parallel_wikipedia':
    sentenceIterator = ParallelSentences
    fileReader = ParallelDataReader
    sentenceReader = GigawordSentenceReader
    wordType = 'words'
    
#read in optional markers file
if args.matches:
    matcher = Matcher(args.matches)
else:
    matcher = None

#read in optional filter file
if args.filter:
    with open(args.filter) as f:
        filterFiles = set(f.read().splitlines())
else:
    filterFiles = None

logging.basicConfig(level=logging.INFO)
    
si = sentenceIterator(args.infiles,
                      fileReader,
                      sentenceReader,
                      sentenceType,
                      args.timeout,
                      args.max,
                      matcher,
                      filterFiles,
                      pairs,
                      labels,
                      verbosity=args.verbosity)

if args.buildType == 'wtmf':
    model = wtmf.WTMFVectorizer(input='content',
                                k=args.n_components,
                                tokenizer=lambda x:x,
                                verbose=True,
                                tf_threshold=args.mc)
    model.fit(si, n_jobs=args.n_jobs)

    #cProfile.runctx("wtmf.build_and_fit(si, args.mc)", globals(), locals(), "Profile.prof")

    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()

    Q = model.transform(si)
    save_params = [Q]

elif args.buildType == 'nmf':
    model = TfkldFactorizer(args.weights, max_iter=args.max_iter, n_components=args.n_components)
    Q = model.fit_transform(si)
    save_params = [Q]
else:
    model = toVec(size=args.n_components, min_count=args.mc, workers=args.n_jobs) #workers=12)
    training = list(si)
    model.build_vocab(training)

    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes

    try:
        for epoch in range(passes):
            model.alpha, model.min_alpha = alpha, alpha
            model.train(training)
        
            print('completed pass %i at alpha %f' % (epoch + 1, alpha))
            alpha -= alpha_delta

            np.random.shuffle(training)
    except KeyboardInterrupt:
        print('terminating on keyboard interrupt')

    save_params = []
    
outfilename = time.time()
if args.outfile:
    outfilename = args.outfile
modelName = "model.{}.{}.{}.{}".format(outfilename, args.buildType, args.sentence, wordType)

model.save('/local/nlp/chidey/' + modelName, *save_params)
#si.save('/local/nlp/chidey/' + modelName + '.titles.bz2')


