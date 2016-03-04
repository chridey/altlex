import sys
import json
import gzip
import os

inputDir = sys.argv[1]
outputFile = sys.argv[2]
maxArticles = int(sys.argv[3])

parallelWikis = {'titles': ['' for i in range(maxArticles)],
                 'files': ['en', 'simple'],
                 'articles': [[[] for i in range(maxArticles)],
                              [[] for i in range(maxArticles)]]}

for filename in os.listdir(inputDir):
    if not filename.endswith('.gz'):
        continue
    print(filename)
    with gzip.open(os.path.join(inputDir,filename)) as f:
        j = json.load(f)

    for article in j:
        if 'title' in article:
            parallelWikis['titles'][article['index']] = article['title']

        for index in range(2):
            sentences = []
            for sentence in article['sentences'][index]:
                newSentence = []
                for partialSentence in sentence['words']:
                    newSentence += partialSentence
                sentences.append(' '.join(newSentence))
            parallelWikis['articles'][index][article['index']] = sentences
        
with gzip.open(outputFile, 'w') as f:
    json.dump(parallelWikis, f)

    
