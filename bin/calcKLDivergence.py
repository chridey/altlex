#for each file, create counts of words appearing for that connective and then not on both sides

import os
import sys
import collections
import math
import itertools

from chnlp.misc import wikipedia

import nltk

stemmer = nltk.PorterStemmer()

def main(dirname, word_pairs=False):
    counts = {'all':
              {'in_s1': collections.defaultdict(int),
               'in_s2': collections.defaultdict(int),
               'in_s2_not_in_s1': collections.defaultdict(int),
               'total': 0}
              }
    discourse = wikipedia.loadDiscourse()
    for filename in os.listdir(dirname):
        if filename in ('.errors', '.likelihoodratio') or filename.startswith('.kld'):
            continue
        counts[filename] = {'in_s1': collections.defaultdict(int),
                            'in_s2': collections.defaultdict(int),
                            'in_s2_not_in_s1': collections.defaultdict(int),
                            'total': 0}
        with open(filename) as f:
            for line in f:
                try:                
                    s1, s2, score = line.strip().decode('utf-8').split('\t')
                except ValueError:
                    print('Problem with line {}'.format(line.decode('utf-8')))
                    continue
                
                if word_pairs:
                    clause1, relation, clause2 = wikipedia.splitOnDiscourse(s1, discourse) #{tuple(filename.split('_'))})
                    stems11 = {'1_' + stemmer.stem(i.lower()) for i in clause1}
                    stems12 = {'2_' + stemmer.stem(i.lower()) for i in clause2}
                    
                    sent1, sent2 = nltk.sent_tokenize(s2)
                    stems21 = {'1_' + stemmer.stem(i.lower()) for i in nltk.word_tokenize(sent1)}
                    stems22 = {'2_' + stemmer.stem(i.lower()) for i in nltk.word_tokenize(sent2)}

                    stems1 = set() #stems11 | stems12 
                    stems1 |= set(itertools.product(stems11, stems12))
                    stems2 = set() #stems21 | stems22 
                    stems2 = set(itertools.product(stems21, stems22))
                else:
                    stems1 = [stemmer.stem(i.lower()) for i in nltk.word_tokenize(s1)]
                    stems1 = set(stems1) | set(zip(stems1, stems1[1:]))
                    stems2 = [stemmer.stem(i.lower()) for i in nltk.word_tokenize(s2)]
                    stems2 = set(stems2) | set(zip(stems2, stems2[1:]))
                
                #count all the words that appear on the right side that don't appear on the left
                for stem in stems2-stems1:
                    counts[filename]['in_s2_not_in_s1'][stem] += 1
                    counts['all']['in_s2_not_in_s1'][stem] += 1    
                #count the total number of times each word appears on the left side and subtract from the total documents    
                for stem in stems1:
                    counts[filename]['in_s1'][stem] += 1
                    counts['all']['in_s1'][stem] += 1
                counts[filename]['total'] += 1
                counts['all']['total'] += 1
                #finally, count just the right-hand side
                for stem in stems2:
                    counts[filename]['in_s2'][stem] += 1
                    counts['all']['in_s2'][stem] += 1

    kld = {}
    kld2 = {}
    print('all: ', counts['all']['total'])
    for filename in counts:
        if filename == 'all':
            continue
        kld[filename] = {'p': {},
                         'q': {},
                         'kl_pq': {}}
        kld2[filename] = {'p': {},
                         'q': {},
                         'kl_pq': {}}
        for stem in counts[filename]['in_s2_not_in_s1']:
            not_in_s1 = counts[filename]['total'] - counts[filename]['in_s1'].get(stem, 0)
            p = 1. * counts[filename]['in_s2_not_in_s1'][stem] / not_in_s1
            all_not_in_s1 = counts['all']['total'] - counts['all']['in_s1'].get(stem, 0) - counts[filename]['total'] + counts[filename]['in_s1'].get(stem, 0) #just added -counts[filename], make sure we dont subtract the connective and s1 twice
            q = 1. * (counts['all']['in_s2_not_in_s1'][stem] - counts[filename]['in_s2_not_in_s1'][stem] + 1) / all_not_in_s1 #just added -counts[filename]

            '''
            if q == 0:
                kl_pq = -float('inf')
            else:
                kl_pq = -p/q
            '''
            
            if p <= 0 or p >= 1 or q <= 0 or q >= 1:
                kl_pq = float('inf')
            else:
                kl_pq = p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))
            #print(stem, p, q, kl_pq)

            kld[filename]['p'][stem] = p
            kld[filename]['q'][stem] = q
            kld[filename]['kl_pq'][stem] = kl_pq

        for stem in counts[filename]['in_s2']:
            #percentage of sentences that have this feature given this connective
            p = 1. * counts[filename]['in_s2'][stem] / counts[filename]['total']
            #percentage of sentences that have this feature given not this connective
            q = 1. * (counts['all']['in_s2'][stem] - counts[filename]['in_s2'][stem] + 1) / (counts['all']['total'] - counts[filename]['total'])
            
            if p <= 0 or p >= 1 or q <= 0 or q >= 1:
                kl_pq = float('inf')
            else:
                kl_pq = p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))
                
            kld2[filename]['p'][stem] = p
            kld2[filename]['q'][stem] = q
            kld2[filename]['kl_pq'][stem] = kl_pq

        print('connective: ', filename, 'counts: ', counts[filename]['total'])
        for stem,score in sorted(kld[filename]['kl_pq'].iteritems(), key=lambda x:x[1], reverse=True):
            print(stem, kld[filename]['p'][stem], kld[filename]['q'][stem], kld[filename]['kl_pq'][stem])
        print('connective2: ', filename, 'counts: ', counts[filename]['total'])
        for stem,score in sorted(kld2[filename]['kl_pq'].iteritems(), key=lambda x:x[1], reverse=True):
            print(stem, kld2[filename]['p'][stem], kld2[filename]['q'][stem], kld2[filename]['kl_pq'][stem])
if __name__ == '__main__':
    word_pairs = False
    if (len(sys.argv) > 2 and sys.argv[2] == '1'):
        word_pairs=True
    print(word_pairs)
    main(sys.argv[1], word_pairs)
