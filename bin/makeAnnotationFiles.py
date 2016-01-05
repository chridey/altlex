from __future__ import print_function

#read from parallel data
#output 100 articles from each letter of the alphabet
#create directory [letter]
#create [articlename].[en|simple].txt in dir

import sys
import collections
import os

import nltk

from chnlp.misc import wikipedia
from chnlp.word2vec import sentenceRepresentation

if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)
    modelFilename = sys.argv[1]
    wikiFilename = sys.argv[2]
    sentRep = sentenceRepresentation.PairedSentenceEmbeddingsClient(wikiFilename,
                                                                    modelFilename)

    discourse = wikipedia.loadDiscourse(format=2)
    articleCounts = collections.defaultdict(int)
    baseDir = 'wikipedia_annotations'
    if not os.path.exists(baseDir):
        os.mkdir(baseDir)
    for articleIndex,title in enumerate(sentRep.iterTitles()):
        if ord(title[0].lower()) not in set(range(97, 123)) or articleCounts[title[0]] > 100:
            continue
        articleCounts[title[0]] += 1

        if not os.path.exists(os.path.join(baseDir, title[0])):
            os.mkdir(os.path.join(baseDir, title[0]))
        for fileIndex,name in enumerate(('en', 'simple')):
            sentences = sentRep.getSentences(articleIndex, fileIndex, pairs=False)
            with open(os.path.join(os.path.join(baseDir,
                                                title[0]),
                                   '{}.{}'.format(articleIndex,
                                                  name)),
                      'w') as f:
                print('Title: {}'.format(title.encode('utf-8')), file=f)
                for sentenceIndex,sentence in enumerate(sentences):
                    words = nltk.word_tokenize(sentence)
                    if len(words) < 4 or ('Template' in sentence and 'has been incorrectly substituted' in sentence):
                        print(file=f)
                        continue

                    #go through one character at a time
                    '''
                    finalString = ''
                    currWord = ''
                    currStart = 0
                    currDiscourse = discourse
                    print(sentence.encode('utf-8'))
                    for i,c in enumerate(sentence.lower()):
                        print('{}|{}|{}|{}|{}'.format(i,
                                                      c.encode('utf-8'),
                                                      currStart,
                                                      currWord.encode('utf-8'),
                                                      finalString.encode('utf-8')))
                        #if c is anything but a lowercase letter
                        if ord(c) in (set(range(32, 126)) - set(range(97, 123))):
                            #if we match a word in the trie, just update the node pointer & continue
                            if currWord in currDiscourse:
                                currDiscourse = currDiscourse[currWord]
                                if currDiscourse == discourse:
                                    currStart = i-len(currWord)
                            else:
                                #then we may have already matched some discourse connective
                                if currDiscourse != discourse and  None in currDiscourse:
                                    finalString += '{}***{}***'.format(sentence[:currStart],
                                                                       sentence[currStart:i])
                                currDiscourse = discourse
                            currWord = ''
                        else:
                            currWord += c
                    finalString += sentence[currStart:]
                    print(finalString.encode('utf-8'))
                    #wikipedia.splitOnDiscourse(sentence, discourse)
                    '''
                    finalString = sentence
                    print('{}: {}'.format(sentenceIndex, finalString.encode('utf-8')), file=f)
                    
                    
#ignore sentences with 'template has been modified' or whatever and less than 4 long
#for each sentence, if there is an explicit discourse connective, mark with ***connective***

#annotation instructions:
#mark each paraphrase pair with a number in the appropriate file ie #1.0.0# the two binary flags indicate whether 1) the sentence is a partial match and 2) whether a) one half has an explicit discourse connective, b) the second half has a different/no discourse connective that changes the discourse relation 

#0: A (named "a" , plural "aes") is the 1st letter and the first vowel in the ISO basic Latin alphabet.
#1: A is the first letter of the English alphabet.
#2: The small letter, a, is used as a lower case vowel.
