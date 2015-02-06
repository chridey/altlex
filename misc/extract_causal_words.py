import sys
import os
from collections import defaultdict
import re
import math

from nltk.stem import SnowballStemmer

r = re.compile('^\s+$')

if __name__ == '__main__':
    sn = SnowballStemmer('english')

    with open('/home/chidey/PDTB/chnlp/config/anticausal_frames') as f:
        anticausatives = {i.lower() for i in f.read().splitlines()}
    
    counts = {}
    for pos in ('nn', 'vv', 'rb', 'jj'):
        counts[pos] = defaultdict(int)
        counts[pos + '_causal'] = defaultdict(int)
        counts[pos + '_anticausal'] = defaultdict(int)
    
    #look for nn/vv/rb/jj
    directory = sys.argv[1]
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

                pos = cols[3].lower()
                if pos.startswith('n'): #'nn' in pos or 'np' in pos:
                    pos = 'nn'
                elif pos.startswith('v'): #'vv' in pos or 'vb' in pos:
                    pos = 'vv'
                elif 'rb' in pos:
                    pos = 'rb'
                elif 'jj' in pos:
                    pos = 'jj'
                else:
                    continue

                for role in cols[11:]:
                    role = role.lower()
                    
                    if 'caus' in role or role in ('reason', 'explanation', 'effect', 'trigger') or 'purpose' in role or 'required' in role or 'consequence' in role or 'result' in role or 'response' in role or 'enabled' in role:
                        counts[pos + '_causal'][word] += 1
                        break

                    if role in anticausatives:
                        counts[pos + '_anticausal'][word] += 1
                        break
                    
                counts[pos][word] += 1
                #print(word, pos, role)

    for pos in ('nn', 'vv', 'rb', 'jj'):
        total = 0
        total_causal = 0
        print(pos)
        with open(pos, 'w') as f:
            for word in sorted(counts[pos], key=lambda x:counts[pos+'_causal'][x]*1.0/counts[pos][x]*math.log(counts[pos][x]), reverse=True):
                if counts[pos][word] > 0:
                    total = counts[pos][word]
                    totalCausal = counts[pos + '_causal'][word]
                    fraction = totalCausal*1.0/total
                    print(pos, word, total, totalCausal, fraction, fraction*math.log(total), file=f)
                    total += counts[pos][word]
            print(sum(counts[pos][word] for word in counts[pos]))
            print(sum(counts[pos + '_causal'][word] for word in counts[pos + '_causal']))

    for pos in ('nn', 'vv', 'rb', 'jj'):
        total = 0
        total_causal = 0
        print(pos)
        with open(pos + '_anticausal', 'w') as f:
            for word in sorted(counts[pos], key=lambda x:counts[pos+'_anticausal'][x]*1.0/counts[pos][x]*math.log(counts[pos][x]), reverse=True):
                if counts[pos][word] > 0:
                    total = counts[pos][word]
                    totalAnticausal = counts[pos + '_anticausal'][word]
                    fraction = totalAnticausal*1.0/total
                    print(pos, word, total, totalAnticausal, fraction, fraction*math.log(total), file=f)
                    total += counts[pos][word]
            print(sum(counts[pos][word] for word in counts[pos]))
            print(sum(counts[pos + '_anticausal'][word] for word in counts[pos + '_anticausal']))
    
