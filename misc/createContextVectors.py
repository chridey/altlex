import sys
import os
import math
import json
import time

from collections import defaultdict, Counter

import numpy

from nltk.stem import SnowballStemmer

from chnlp.utils.readers.parsedGigawordReader import ParsedGigawordReader
from chnlp.utils.readers.sentenceReader import SentenceReader

sn = SnowballStemmer('english')

r = ParsedGigawordReader(sys.argv[1])

context = defaultdict(Counter)
before = ['START{}'.format(i) for i in range(4)]
after = ['END{}'.format(i) for i in range(4)]

starttime = time.time()
timeout = 60*60*168
try:
    for s in r.iterFiles():
        sr = SentenceReader(s)

        for sentence in sr.iterSentences(False):
            stems = before + sentence.stems + after
            for index,stem in enumerate(stems[4:-4]):
                for sim in stems[index-4:index+4]:
                    context[stem][sim] += 1
        
        if time.time() - starttime > timeout:
            raise Exception

except KeyboardInterrupt:
    pass
except Exception:
    pass

with open(sys.argv[2] + '_context.json', 'w') as f:
    json.dump(context, f)

