import sys
import json

from collections import defaultdict

#from sklearn.feature_extraction.text import TfidfTransformer

from chnlp.ml.affinityPropagation import AffinityPropagator
from chnlp.ml.tfidfTransformer import TfIdf

if sys.version_info < (3,):
    from mrec.mf.wrmf import WRMFRecommender

with open(sys.argv[1]) as f:
    j = json.load(f)

test = []
lookup = {}
for index,feature in enumerate(j):
    test.append(j[feature])
    lookup[index] = feature

ap = AffinityPropagator()
tfidf = TfIdf()

print("here")
x = tfidf._transform(test)
print(len(x), len(x[0]))
x = tfidf.classifier.fit_transform(x)
print("done tfidf")

wrmfr = WRMFRecommender(100)

wrmfr.fit(x)

print("clustering")
ap.train(wrmfr.U, False)
ap.featureMap = lookup

#with open(sys.argv[1] + '.lookup', 'w') as f:
#    json.dump(lookup, f)

ap.save(sys.argv[1] + '.model')

s = defaultdict(list)
for i,v in enumerate(ap.classifier.labels_):
    s[lookup[v]].append(featureMap[i])

with open(sys.argv[1] + '.clusters.json', 'w') as f:
    json.dump(s, f)
