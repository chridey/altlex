import sys

import numpy
import scipy
import json

from mrec.mf.wrmf import WRMFRecommender

wrmfr = WRMFRecommender(100)

a = numpy.load(sys.argv[1])
s = scipy.sparse.csr_matrix(a)

wrmfr.fit(s)

print(type(wrmfr.U))

#wrmfr.U.dump(sys.argv[1] + '_wtmf')

with open(sys.argv[1] + '_wtmf', 'w') as f:
    json.dump(wrmfr.U, f)
