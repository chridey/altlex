import math
import sqlite3
import sys
from collections import defaultdict

import chnlp.pdtb.pdtbsql as p

cb = p.CorpusBuilder(sys.argv[1])
c = cb.extract()

total_sentences = len(c.di)
total_causal = len([i for i in c.di if '.Cause.' in i.klass])
print(total_sentences, total_causal)

#now count how often each word appears 
total_word_appearances = defaultdict(int)
total_causal_word_appearances = defaultdict(int)
for i in c.di:
    x = set(i.all_words)
    for w in x:
        total_word_appearances[w] +=1
        if '.Cause.' in i.klass:
            total_causal_word_appearances[w] += 1

#now do the stats
stats = {}
for w in total_word_appearances:
    tf = total_causal_word_appearances[w]
    idf = math.log(total_sentences
    stat = (1.0*total_causal_word_appearances[w]/total_word_appearances[w])/(1.0*total_causal/total_sentences)
    stats[w] = stat

for s in sorted(stats, key=stats.get, reverse=True):
    print(s,stats[s])
    
