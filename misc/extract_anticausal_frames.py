import sys
import os
from collections import defaultdict
import re
import math

from nltk.stem import SnowballStemmer

r = re.compile('^\s+$')

if __name__ == '__main__':
    sn = SnowballStemmer('english')

    with open(sys.argv[1]) as f:
        anticausativeWords = {sn.stem(i.lower()) for i in f.read().splitlines()}
    anticausativeFrames = set()
    
    directory = sys.argv[2]
    files = os.listdir(directory)
    for fi in files:
        with open(directory + fi) as f:
            print(fi)
            for line in f:
                if r.match(line) or line.startswith('SKIPPED') or line.startswith('#'):
                    continue
                cols = line.split()
                #print(line, cols)
                word = sn.stem(cols[2].lower()) #cols[2]
                if word in anticausativeWords:
                    for role in cols[11:]:
                        if role != '_':
                            anticausativeFrames.add(role)
                        
for frame in anticausativeFrames:
    print(frame)
