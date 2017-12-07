from __future__ import print_function

import json
import sys

from altlex.utils.readers.plaintextIterator import TrainSetIterator
from altlex.utils.readers.plaintextIterator import TestSetIterator

def process_file(filename, iterator_type, kwargs):
    iterator = iterator_type(filename, **kwargs)
    outfilename = filename.split('.')[0] + '.jsonlist'
    with open(outfilename, 'w') as f:
        for metadata in iterator:
            print(json.dumps(metadata), file=f)

train_file = sys.argv[1]
dev_file = sys.argv[2]
test_file = sys.argv[3]

kwargs = {'frames': True}
process_file(train_file, TrainSetIterator, kwargs)
process_file(dev_file, TestSetIterator, kwargs)
process_file(test_file, TestSetIterator, kwargs)
