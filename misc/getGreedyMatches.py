import sys
import collections

filename = sys.argv[1]
minDoc2Vec = float(sys.argv[2])
minWiknet = float(sys.argv[3])
penalty = float(sys.argv[4])

articleLookup = collections.defaultdict(list)
with open(filename) as f:
    for line in f:
        articleIndex, sentence1, sentence2, doc2Vec, wiknet, unmatched = line.strip().split('\t')
        if sentence1[0] == '(':
            _index1, _index2 = sentence1[1:-1].split(',')
            index1 = (int(_index1),int(_index2))
        else:
            index1 = (int(sentence1),)

        if sentence2[0] == '(':
            _index1, _index2 = sentence2[1:-1].split(',')
            index2 = (int(_index1),int(_index2))
        else:
            index2 = (int(sentence2),)

        doc2Vec = float(doc2Vec)
        wiknet = float(wiknet)
        unmatched = float(unmatched)
        wiknetPenalized = wiknet-penalty*unmatched
        if doc2Vec < minDoc2Vec:
            continue
        if wiknetPenalized < minWiknet:
            continue

        harmonicMean = 2*doc2Vec*wiknetPenalized/(doc2Vec+wiknetPenalized)
        
        articleLookup[int(articleIndex)].append([index1, index2, harmonicMean])
print('done reading file')

for articleIndex in sorted(articleLookup.keys()):
    foundIndices1 = set()
    foundIndices2 = set()
    for index1, index2, harmonicMean in sorted(articleLookup[articleIndex],
                                               reverse=True,
                                               key=lambda x:x[-1]):
        if index1[0] in foundIndices1 or index2[0] in foundIndices2 or len(index1) > 1 and index1[1] in foundIndices1 or len(index2) > 1 and index2[1] in foundIndices2:
            continue
        foundIndices1.update(set(index1))
        foundIndices2.update(set(index2))
        print('{}\t{}\t{}\t{}'.format(articleIndex, index1, index2, harmonicMean))


