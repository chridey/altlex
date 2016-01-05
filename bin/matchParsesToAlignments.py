#go through the original aligned files and parses simultaneously
#since there are fewer parses (because of the min length restriction) we increment the pointer in the aligned files if the parse does not match
#output the results to one of 5 files

#new strategy
#read in all the parse files, sort uniq them by sentence
#read in all the alignment files, sort uniq them by sentence
#ignore any alignment files > 55 in length, or ending in not ending punctuation, or more than one sentence

import sys
import os
import operator

import nltk
stemmer = nltk.PorterStemmer()

import re
#cornerCaseRe = re.compile('^(.+ [A-Z]\. )(.+)$')
#cornerCaseRe2 = re.compile('^(.+[\x80-\xff].\. )(.+)$')
cornerCases = [re.compile('^(.+? [A-Za-z]\. )(.+ [A-Za-z]\. )?(.+)$')]
                          
def getParseWords(parseString):
    parseWords = nltk.Tree.fromstring(parseString).leaves()
    parseSent = ' '.join(parseWords)
    return nltk.word_tokenize(parseSent.replace('-LRB-', '(').replace('-RRB-', ')').replace('#', 'HASH').decode('utf-8'))

def getAlignmentSent(alignmentString):
    return alignmentString.split('\t')[0].replace(' \\/ ', ' \\ / ').replace('#', 'HASH').replace('http:\\/\\/', 'http : \\ / \\ / ').decode('utf-8')

def getDiscourse(parseWords):
    disc = {'0'}
    #get their class
    #output to appropriate class file in output dir

    for word in parseWords:
        if 'HASH' in word:
            markup = word.split('HASH')
            if len(markup) == 3:
                word, wid, wordClass = markup
                disc.add(wordClass)

    return disc
    
if __name__ == '__main__':
    alignedDir = sys.argv[1]
    parseDir = sys.argv[2]
    outputDir = sys.argv[3]
    finalAlignments = {}
    finalParses = {}
    for connective in os.listdir(alignedDir):
        if connective.startswith('.'):
            continue
        print(connective)
        connectiveFile = os.path.join(alignedDir, connective)
        parseFile = os.path.join(parseDir, connective) + '.sentonly.parsed.disc'

        if not os.path.exists(parseFile):
            print('No file {}'.format(parseFile))
            continue

        with open(connectiveFile) as f:
            for line in f:
                line = line.strip()
                alignment = getAlignmentSent(line)
                alignmentWords = nltk.word_tokenize(alignment)
                finalAlignments[' '.join(alignmentWords)] = line

        with open(parseFile) as f:
            for line in f:
                parseWords = getParseWords(line)
                finalParseWords = []
                for word in parseWords:
                    s = word.split('HASH')
                    if len(s) == 3:
                        finalParseWords.append(s[0])
                    else:
                        finalParseWords.append(word)
                finalParses[' '.join(finalParseWords)] = line

        #break

    '''
    print(len(finalParses), len(finalAlignments))
    s1 = sorted(finalParses.items(), key=lambda x:x[0])
    s2 = sorted(finalAlignments.items(), key=lambda x:x[0])
    for i in range(10):
        print (s1[i])
        print (s2[i])
    '''
    
    matches = set(finalParses.keys()) & set(finalAlignments.keys())
    print(len(matches))
    for match in matches:
        parseWords = getParseWords(finalParses[match])
        disc = getDiscourse(parseWords)
        for d in disc:
            if d == '0' and len(disc) > 1:
                continue
            with open(os.path.join(outputDir, d), 'a') as f:
                f.write(finalAlignments[match])
                f.write("\n")
