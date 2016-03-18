from __future__ import print_function

import os
import sys
import gzip
import json

indir = sys.argv[1]
outfile1 = sys.argv[2]
outfile2 = sys.argv[3]

if len(sys.argv) > 4:
    maxItems = int(sys.argv[4])
else:
    maxItems = float('inf')
    
f1 = open(outfile1, 'w')
f2 = open(outfile2, 'w')

count = 0
for filename in sorted(os.listdir(indir)):
    if not filename.endswith('.gz'):
        continue
    print(filename)
    with gzip.open(os.path.join(indir,filename)) as f:
        j = json.load(f)
    print(len(j))
    assert(len(j) % 2 == 0)
    for index,i in enumerate(j):
        lemmas = []
        for lemmas_partial in i['lemmas']:
            lemmas += [x.lower() for x in lemmas_partial]
        
        if index % 2 == 0:
            print(' '.join(lemmas).encode('utf-8'), file=f1)
        else:
            print(' '.join(lemmas).encode('utf-8'), file=f2)
            count += 1
        if count > maxItems:
            break
    if count > maxItems:
            break
        

f1.close()
f2.close()
