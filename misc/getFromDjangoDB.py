import sqlite3
import json
import sys

from chnlp.altlex.dataPoint import DataPoint

from chnlp.metrics.kappa import fleiss_kappa

import numpy as np

conn = sqlite3.connect(sys.argv[2])
curs = conn.cursor()
#conn.text_factory = bytes        

curs.execute('select annotator_id,real_id,metadata,tag from causality_relation r, causality_taggedrelation tr where r.id = tr.relation_id and tr.annotator_id = 1 and tr.relation_id between 1 and 486  union select annotator_id,real_id,metadata,tag from causality_relation r, causality_taggedrelation tr where r.id = tr.relation_id and tr.annotator_id = 2 and tr.relation_id between 487 and 975 union select annotator_id,real_id,metadata,tag from causality_relation r, causality_taggedrelation tr where r.id = tr.relation_id and tr.annotator_id = 3 and tr.relation_id between 976 and 1463 union select annotator_id,real_id,metadata,tag from causality_relation r, causality_taggedrelation tr where r.id=tr.relation_id and tr.annotator_id in (1,2,3) and tr.relation_id > 1463 order by real_id')
#in (1,2,3)

lookup = {}
for row in curs:
    annotator_id, real_id, metadata, tag = row
    print(real_id, annotator_id)
    if real_id not in lookup:
        lookup[real_id] = (json.loads(metadata), [tag])
    else:
        lookup[real_id][1].append(tag)

j = []
tags = ('causal', 'notcausal')
prevTag = 0
causalCount = 0
agree = 0
disagree = 0
examples = []
iaaCount = 0
causalChanged = 0
nonCausalChanged = 0

for real_id in sorted(lookup):
    #print(real_id)
    metadata, tags = lookup[real_id]

    if len(tags)-sum(tags) < sum(tags):
        causalCount+=1
        tag = 'causal'
    elif len(tags)-sum(tags) == sum(tags):
        tag = tags[prevTag]
        prevTag ^= 1
    else:
        tag = 'notcausal'

    #calculate IAA
    if len(tags) > 1:
        iaaCount += 1
        tagCount = [0,0]
        
        print(tags)
        if sum(tags) == len(tags) or sum(tags) == 0:
            agree +=1
        else:
            disagree += 1
        for i in range(len(tags)):
            tagCount[tags[i]] += 1

        for i in range(len(tagCount)):
            if (tagCount[i]):
                examples.extend([(iaaCount,i)] * tagCount[i])

    if metadata['tag'] == 'causal' and tag != 'causal':
        causalChanged+=1

    if metadata['tag'] != 'causal' and tag == 'causal':
        nonCausalChanged+=1
        
    metadata['tag'] = tag
    j.append(metadata)

print(causalCount, len(j))
print(causalChanged, nonCausalChanged)

total = agree+disagree
print(agree, disagree, agree+disagree, agree/(agree+disagree))

print(examples)
print(fleiss_kappa(examples, 3, 2))

conn.close()

with open(sys.argv[1], 'w') as f:
    json.dump(j, f)
