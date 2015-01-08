import sys
from collections import defaultdict

from nltk.corpus.reader.wordnet import WordNetError

import chnlp.pdtb.pdtbsql as p
import chnlp.misc.word_pmi as wp

cb = p.CorpusBuilder(sys.argv[1])
c = cb.extract(relation='AltLex', klass='%.Cause.%')
x = defaultdict(int)

for di in c.di:
    for w in set(di.connective):
        x[w] += 1

for s in sorted(x, key=x.get, reverse=True):
    print(s,x[s])
    
if len(sys.argv) <= 2:
    exit()
#now do PMI for each of these
#wp.word_pmi(sys.argv[1], x)

#just do something dumb, if it has one of those words, predict true, otherwise, predict false
cb = p.CorpusBuilder(sys.argv[1])
c = cb.extract()
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
set_x = set(i for i in x if x[i] >= 20)
for di in c.di:
    if di.relation in ('AltLex', 'Explicit') and '.Cause.' in di.klass:
        continue

    #"smarter" - try path similarity
    
    if set_x & set(di.secondarg):
        prediction = True
    else:
        prediction = False

    if prediction:
        if '.Cause.' in di.klass:
            true_positives += 1
        else:
            false_positives += 1
    else:
        if '.Cause.' in di.klass:
            false_negatives += 1
        else:
            true_negatives += 1

print("true_positives: {}", true_positives)
print("false_positives: {}", false_positives)
print("true_negatives: {}", true_negatives)
print("false_negatives: {}", false_negatives)

print("accuracy: {}", (true_positives+true_negatives)/(true_positives+true_negatives+false_negatives+false_positives))
