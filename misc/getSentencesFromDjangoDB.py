import sqlite3
import json
import sys

from chnlp.altlex.dataPoint import DataPoint

conn = sqlite3.connect(sys.argv[1])
curs = conn.cursor()
#conn.text_factory = bytes        

curs.execute('select sentence1,sentence2 from causality_relation r order by real_id')

for row in curs:
    print(row[0])
    print(row[1])

conn.close()

