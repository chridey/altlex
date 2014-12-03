import sys
import os
from collections import defaultdict
import re
r = re.compile('^\s+$')

if __name__ == '__main__':
    counts = {}
    for pos in ('nn', 'vv', 'rb', 'jj'):
        counts[pos] = defaultdict(int)
        counts[pos + '_causal'] = defaultdict(int)
    
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
                word = cols[2]
                pos = cols[3].lower()
                role = cols[11].lower()
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
                
                if 'caus' in role or role in ('reason', 'explanation', 'effect', 'trigger') or 'purpose' in role or 'required' in role or 'consequence' in role or 'result' in role or 'response' in role or 'enabled' in role:
                    counts[pos + '_causal'][word] += 1
                counts[pos][word] += 1
                #print(word, pos, role)

    for pos in ('nn', 'vv', 'rb', 'jj'):
        total = 0
        total_causal = 0
        print(pos)
        with open(pos, 'w') as f:
            for word in sorted(counts[pos], key=lambda x:counts[pos+'_causal'][x]*1.0/counts[pos][x], reverse=True):
                counts[pos][word] += 1
                print(pos, word, counts[pos][word], counts[pos + '_causal'][word], counts[pos + '_causal'][word]*1.0/counts[pos][word], file=f)
                total += counts[pos][word]
            print(sum(counts[pos][word] for word in counts[pos]))
            print(sum(counts[pos + '_causal'][word] for word in counts[pos + '_causal']))
    
