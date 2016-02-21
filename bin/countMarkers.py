#script to count the occurrences of all the markers in the PDTB

import os
import sys
import collections

from nltk import word_tokenize

from chnlp.utils import wordUtils

all_markers = [list(i) for i in wordUtils.causal_markers | wordUtils.all_markers | wordUtils.noncausal_markers]

wsj_dir = sys.argv[1]
counts = collections.defaultdict(int)

for wsj_subdir in os.listdir(wsj_dir):
    full_wsj_subdir = os.path.join(wsj_dir, wsj_subdir)
    print(full_wsj_subdir)
    for filename in os.listdir(full_wsj_subdir):
        print(filename)
        with open(os.path.join(full_wsj_subdir, filename)) as f:
            for line in f:
                words = word_tokenize(line.strip())
                for i in range(len(words)):
                    for x in all_markers:
                        if words[i:i+len(x)] == x:
                            counts[tuple(x)] += 1


for key in sorted(counts, key=lambda x:counts[x]):
    print(key, counts[key])
