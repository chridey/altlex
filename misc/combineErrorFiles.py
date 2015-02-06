import json
import sys

j = []
for f in sys.argv[1:]:
    with open(f) as fp:
        j += json.load(fp)

json.dump(j, sys.stdout)
