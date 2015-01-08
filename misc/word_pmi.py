import sqlite3
import sys
from collections import defaultdict

import chnlp.pdtb.pdtbsql as p
from chnlp.semantics.wordNetManager import WordNetManager

def word_pmi(corpus, words=None):

    cb = p.CorpusBuilder(corpus)
    c = cb.extract()

    total_sentences = len(c.di)
    total_causal = len([i for i in c.di if '.Cause.' in i.klass])
    print(total_sentences, total_causal)

    #now count how often each word appears 
    total_word_appearances = defaultdict(int)
    total_causal_word_appearances = defaultdict(int)
    for i in c.di:
        #x = set(i.all_bigrams)
    #x = set(i.all_words)
    #x = set(i.firstarg)
    #x = [' '.join(i.connective)]
        x = set(i.secondarg)
        for w in x:
            if words is not None and w not in words:
                continue
            total_word_appearances[w] +=1
            if '.Cause.' in i.klass:
                total_causal_word_appearances[w] += 1

    #read in the altlexes
    altlexes = {'cause', 'result', 'reason', 'mean'} #set()
    '''
    with open("altlexes") as f:
        for altlex in f.readlines():
            word,count = altlex.split()
            if int(count) >= 2:
                altlexes.add(word)
    '''
    
    #now do the stats
    #p(word | causal) / p(word) > 1
    stats = {}
    for w in total_word_appearances:
        if total_word_appearances[w] < 10:
            continue
    #same thing
    #stat = (1.0*total_causal_word_appearances[w]/total_word_appearances[w])/(1.0*total_causal/total_sentences)
        stat = (1.0*total_causal_word_appearances[w]/total_causal)/(1.0*total_word_appearances[w]/total_sentences)
        stats[w] = stat

    wnm = WordNetManager()
    for s in sorted(stats, key=stats.get, reverse=True):
        print(s,stats[s],max(wnm.distance(s,i) for i in altlexes))
    
if __name__ == '__main__':
    word_pmi(sys.argv[1])
