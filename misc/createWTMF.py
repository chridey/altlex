import sys
import math
import json

import numpy
import scipy

from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

if sys.version_info < (3,):
    from mrec.mf.wrmf import WRMFRecommender

idfSetting = int(sys.argv[2])
assert(idfSetting in (0,1))
#0 does document level IDF
#1 does sentence level
matrixDecompositionSetting = int(sys.argv[3])
assert(matrixDecompositionSetting in (0,1))
#0 does SVD
#1 does WTMF

counts = numpy.load(sys.argv[1])
with open(sys.argv[1] + '_lookup.json') as f:
    j = json.load(f)

idf = j['idf']
N = j['N']
if idfSetting == 1:
    idf = j['sentence_idf']
    N = j['sentenceN']

reverseWordLookup = {j:i for (i,j) in j['words'].items()}
sn = SnowballStemmer("english")
stopWords = {sn.stem(i) for i in stopwords.words("english")}
stopWords.update({",", "!", "%", "&", "(", ")", "-", ":", ";", ",", ".", "?"})
#multiply by IDF 
for wordIndex in range(len(counts)):
    if reverseWordLookup[wordIndex] in j['seeds'] or reverseWordLookup[wordIndex] in stopWords:
        thisIDF = 0
    else:
        thisIDF = math.log(N/(idf[reverseWordLookup[wordIndex]]))
    #print(reverseWordLookup[wordIndex], thisIDF)
    for seedIndex in range(len(counts[wordIndex])):
        counts[wordIndex][seedIndex] *= thisIDF

s = scipy.sparse.csr_matrix(counts, dtype=numpy.float64)
print('done')

if matrixDecompositionSetting == 1:
    wrmfr = WRMFRecommender(100)

    wrmfr.fit(s)

    print(type(wrmfr.U))

    #wrmfr.U.dump(sys.argv[1] + '_wtmf')

    with open(sys.argv[1] + '_wtmf', 'w') as f:
        json.dump([wrmfr.U.tolist(),
                   wrmfr.V.tolist()], f)
else:
    tsvd = TruncatedSVD(n_components=100)
    X_new = tsvd.fit_transform(numpy.transpose(s))
    print(X_new.shape)
    #print(X_new)
    print(tsvd.explained_variance_ratio_) 
    joblib.dump(tsvd, sys.argv[1] + '_svd')
    X_new.dump(sys.argv[1] + '_svd.npy')
    '''
    s = numpy.transpose(s)
    print(s.shape)
    print(numpy.rank(s))
    U, sigma, V = scipy.sparse.linalg.svds(s, k=10)
    with open(sys.argv[1] + '_svd.npz', 'w') as f:
        numpy.savez_compressed(f, U=U, sigma=sigma, V=V)
    '''
