import sys
import json
import cjson

with open(sys.argv[1]) as f:
    #j = json.load(f)
    j = cjson.decode(f.read())

final = {}
tags = set()
for marker in j:
    print(marker)
    if marker == sys.argv[2]:
        for lemma1 in j[marker]:
            junk = lemma1.split('-')
            tags.add(junk[-1])
            lemma = '-'.join(junk[:-1])
            
            if '-' + sys.argv[3] in lemma1:
                print(lemma1)
                if lemma not in final:
                    final[lemma] = {}
                for lemma2 in j[marker][lemma1]:
                    junk = lemma2.split('-')
                    tags.add(junk[-1])
                    p = junk[-1]
                    if p in {'DATE', 'DURATION', 'TIME',
                             'MONEY', 'NUMBER', 'PERCENT',
                             'ORDINAL',
                             'PERSON', 'ORGANIZATION', 'LOCATION'}:
                        if p not in final[lemma]:
                            final[lemma][p] = 0
                        final[lemma][p] += j[marker][lemma1][lemma2]
                    else:
                        l = '-'.join(junk[:-1])
                        if l not in final[lemma]:
                            final[lemma][l] = 0
                        final[lemma][l] += j[marker][lemma1][lemma2]

print(len(final))
#for tag in sorted(tags):
#    print(tag)
with open(sys.argv[4] + '.json', 'w') as f:
    json.dump(final, f)
