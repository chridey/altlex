from __future__ import print_function

import sys
import os
import argparse

from chnlp.misc import wikipedia
import nltk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract pairs with connectives')

    parser.add_argument('infiles', 
                        help='the files containing the pairs and their scores',
                        nargs='+')    
    parser.add_argument('outdir', 
                        help='the directory to write the pairs to')
    parser.add_argument('--threshold', type=float, default=.4)
    parser.add_argument('--splitter', default='\t')

    args = parser.parse_args()

    discourse = wikipedia.loadDiscourse('/home/chidey/PDTB/chnlp/config/markers')

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    for infile in args.infiles:
        with open(infile) as f:
            for line in f:
                if line.startswith('Title:') or args.splitter not in line:
                    continue
                s = line.strip().decode('utf-8').split(args.splitter)
                if len(s) != 3:
                    print('Problem with splitting line: {}'.format(line), file=sys.stderr)
                    continue
                sent1, sent2, score = s
                if float(score) < args.threshold:
                    continue
                conn1 = wikipedia.splitOnDiscourse(sent1, discourse, True)
                conn2 = wikipedia.splitOnDiscourse(sent2, discourse, True)
                split1 = nltk.sent_tokenize(sent1)
                split2 = nltk.sent_tokenize(sent2)
                if len(split1) > 2 or len(split2) > 2:
                    print('Problem splitting sentences: {}'.format(line), file=sys.stderr)
                    continue
                if len(split1) == 1: 
                    for conn in conn1-conn2:
                        with open(os.path.join(args.outdir, '_'.join(conn)), 'a+') as f:
                            print('\t'.join((sent1, sent2, score)).encode('utf-8'), file=f)
                if len(split2) == 1: 
                    for conn in conn2-conn1:
                        with open(os.path.join(args.outdir, '_'.join(conn)), 'a+') as f:
                            print('\t'.join((sent2, sent1, score)).encode('utf-8'), file=f)                        
            
