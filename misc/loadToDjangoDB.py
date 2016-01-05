import sqlite3
import json
import sys

from chnlp.altlex.dataPoint import DataPoint

conn = sqlite3.connect(sys.argv[2])
curs = conn.cursor()
conn.text_factory = bytes        

with open(sys.argv[1]) as f:
    j = json.load(f)

for count,datum in enumerate(j):
    dp = DataPoint(datum)
    query = 'INSERT INTO causality_relation (real_id, sentence1, sentence2, metadata) values (?, ?, ?, ?)'
    sentence2 = ' '.join(dp.getCurrWords())
    sentence1 = ' '.join(dp.getPrevWords())
    if sentence1[-2:] == ' .':
        sentence1 = sentence1[:-2] + '.'
    if sentence2[-2:] == ' .':
        sentence2 = sentence2[:-2] + '.'
    print(count, sentence1, sentence2    )
    curs.execute(query, [count, sentence1, sentence2, json.dumps(datum)])

conn.commit()
conn.close()
