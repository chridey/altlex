import sys

from collections import defaultdict
from nltk.corpus.reader.wordnet import WordNetError

import chnlp.pdtb.pdtbsql as p
import chnlp.misc.word_pmi as wp

cb = p.CorpusBuilder(sys.argv[1])
c = cb.extract(relation='AltLex', klass='%.Cause.%', stem=False, lemmatize=False, stop=False)
x = defaultdict(int)

for di in c.di:
    c = ' '.join(di.connective)
    x[c] += 1

for s in sorted(x, key=x.get, reverse=True):
    print(s,x[s])
    
if len(sys.argv) <= 2:
    exit()

cb = p.CorpusBuilder(sys.argv[1])
c = cb.extract(stem=False, lemmatize=False, stop=False)

causal = defaultdict(int)
total = defaultdict(int)
for di in c.di:
    if di.relation == 'Explicit':
        continue
    words = ' '.join(di.secondarg)
    for i in x:
        #if i in words:
        if words.startswith(i):
            if '.Cause.' in di.klass:
                causal[i] += 1
            total[i] += 1

for s in sorted(total, key=lambda s:causal[s]*1.0/total[s], reverse=True):
    if total[s] > 1:
        print(s, x[s], total[s], causal[s], causal[s]*1.0/total[s])
            

