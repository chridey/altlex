#read in sentences and scores
#for each sentence in the parses
#if the parse is in the sentence score lookup, add it to a dictionary
#write the results to a file

import sys
import os
import gzip
import json

import nltk

from chnlp.misc import wiknet

lookup = set()
with open(sys.argv[1]) as f:
    for line in f:
        try:
            sent1, sent2, score = line.decode('utf-8').split('\t') #(' , ')
        except Exception:
            print('problem with {}'.format(line))
            continue
        for sent in (sent1, sent2):
            words = sent.split() #nltk.word_tokenize(sent)
            lookup.add(' '.join(words))

final = {}
for filename in os.listdir(sys.argv[2]):
    if not filename.endswith('.gz'):
        continue
    try:
        with gzip.open(os.path.join(sys.argv[2],filename)) as f:
            j = json.load(f)
    except IOError:
        print('problem with {}'.format(filename))
        continue
    for p in j:
        for wiki in p['sentences']:
            for index,sentence in enumerate(wiki):
                words = wiknet.getLemmas(sentence['words'])
                if ' '.join(words) in lookup:
                    print(filename, p['title'])
                    final[' '.join(words)] = sentence
                if index < len(wiki)-1:
                    words += wiknet.getLemmas(wiki[index+1]['words'])
                    if ' '.join(words) in lookup:
                        print(filename, p['title'])
                        final[' '.join(words)] = sentence
                
print(len(final))

with gzip.open(sys.argv[3], 'w') as f:
    json.dump(final, f)
