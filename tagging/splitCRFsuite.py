import sys
import random
from collections import defaultdict

from chnlp.utils.utils import balance

with open(sys.argv[1]) as f:
    t = f.read().split('\n\n')

with open(sys.argv[1] + '.train', 'w') as train:
    with open(sys.argv[1] + '.test', 'w') as test:
        for s in t:
            if random.random() <= .9:
                print(s + "\n", file=train)
            else:
                print(s + "\n", file=test)

