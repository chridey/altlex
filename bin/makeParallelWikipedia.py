import sys
import json
import gzip

import nltk

import wikipedia
from chnlp.utils.readers.wikipediaReader import ExtractedWikipediaReader

def getWeb(titles, discourse=None):
    titleList = []
    enList = []
    simpleList = []
    for title in titles:
        print(title)
        en, simple = wikipedia.getParallelArticles(title, discourse)
        enList.append(en)
        simpleList.append(simple)
    return {'titles': titleList,
            'en': enList,
            'simple': simpleList}

def getXML(titles, enreader, wikireader):
    titleLists = []
    articleLists = []
    starts = []
    for i,reader in enumerate((enreader, wikireader)):
        titleLists.append([])
        articleLists.append([])
        starts.append([0])
        for j,(name,sentences) in enumerate(reader.iterFiles()):
            print(name)
            if name not in titles:
                continue
            titleLists[i].append(name)

            articleLists[i].append(sentences)
            starts[i].append(starts[i][len(starts[i])-1] + len(sentences))

    titleList = list(sorted(titles))
    enLookup = {j:i for i,j in enumerate(titleLists[0])}
    simpleLookup = {j:i for i,j in enumerate(titleLists[1])}

    enIndex = []
    simpleIndex = []
    finalList = []
    for i,title in enumerate(titleList):
        if title in enLookup and title in simpleLookup:
            enIndex.append(enLookup[title])
            simpleIndex.append(simpleLookup[title])
            finalList.append(title)
            
    return {'files': ['en', 'simple'],
            'titles': finalList,
            'en': {'articles': articleLists[0],
                   'starts': starts[0],
                   'orig': enIndex},
            'simple': {'articles': articleLists[1],
                       'starts': starts[1],
                       'orig': simpleIndex}}

def convert(filename, outfilename):
    with gzip.open(filename) as f:
        data = json.load(f)

    output = {'titles': data['titles'],
              'articles': [],
              'starts': [],
              'files': data['files']}
    
    for filename in data['files']:
        articles = []
        starts = [0]
        for index,title in enumerate(data['titles']):
            origIndex = data[filename]['orig'][index]
            article = data[filename]['articles'][origIndex]
            articles.append(article)
            starts.append(starts[len(starts)-1] + len(article))
        output['articles'].append(articles)
        output['starts'].append(starts)

    with gzip.GzipFile(outfilename, 'w') as f:
        json.dump(output, f)
        
if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]

    if len(sys.argv) == 4:
        convert(infile, outfile)
        exit()

    with open(infile) as f:
        titles = f.read().splitlines()

    if len(sys.argv) > 3:
        enwikifile = sys.argv[3]
        simplewikifile = sys.argv[4]
        enreader = ExtractedWikipediaReader(enwikifile)
        simplereader = ExtractedWikipediaReader(simplewikifile)
        ret = getXML(titles, enreader, simplereader)
    else:
        ret = getWeb(titles)

    with gzip.GzipFile(outfile, 'w') as f:
        json.dump(ret, f)
