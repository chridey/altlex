import sys
from collections import defaultdict

count = defaultdict(int)
with open(sys.argv[1]) as f:
    for line in f:
        length = len(line.split(' '))
        count[length] += 1
        if length > 100:
            print (line)

print (count)
