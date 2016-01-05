#go through the original aligned files and parses simultaneously
#since there are fewer parses (because of the min length restriction) we increment the pointer in the aligned files if the parse does not match
#output the results to one of 5 files

#new strategy
#read in all the parse files, sort uniq them by sentence
#read in all the alignment files, sort uniq them by sentence
#ignore any alignment files > 60 in length, or ending in not ending punctuation, or more than one sentence

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

def getDiscourse(parseWords, alignmentWords, connectiveList):
    match = False
    disc = ['0']
    wordPtr = 0
    #if len(parseWords) == len(alignmentWords):
    #then do secondary check

    match = True
    connectivePtr = 0

    #go through the parsed sentence and check for connectives that match this filename
    #get their class
    #output to appropriate class file in output dir

    for wordPtr in range(min(len(parseWords), len(alignmentWords))):
        #print(alignmentWords[wordPtr], parseWords[wordPtr], alignmentWords[wordPtr] == parseWords[wordPtr])
        if alignmentWords[wordPtr] != parseWords[wordPtr]:
            if 'HASH' in parseWords[wordPtr]:
                markup = parseWords[wordPtr].split('HASH')
                if len(markup) == 3:
                    word, wid, wordClass = markup
                    if word == alignmentWords[wordPtr]:
                        #check if this word is the start of the current connective
                        if  stemmer.stem(word.lower()) == connectiveList[connectivePtr]:
                            connectivePtr += 1
                            if connectivePtr >= len(connectiveList):
                                connectivePtr = 0
                                disc.append(wordClass)
                        continue
            match = False
            break

    if len(parseWords) != len(alignmentWords):
        match = False

    return match, disc, wordPtr
    
if __name__ == '__main__':
    alignedDir = sys.argv[1]
    parseDir = sys.argv[2]
    outputDir = sys.argv[3]
    for connective in os.listdir(alignedDir):
        connectiveFile = os.path.join(alignedDir, connective)
        connectiveList = connective.split('_')
        parseFile = os.path.join(parseDir, connective) + '.sentonly.parsed.disc'
        with open(connectiveFile) as f:
            alignments = f.read().splitlines()
        
        with open(parseFile) as f:
            parses = f.read().splitlines()

        parsePtr = 0
        alignmentPtr = 0
        while alignmentPtr < len(alignments) and parsePtr < len(parses):
            #also need to handle special cases where the parse has been combined because of a : and where the "sentence" in the alignment file is actually two sentences
            parseWords = getParseWords(parses[parsePtr])

            alignmentSent = getAlignmentSent(alignments[alignmentPtr])
            #split sentence here
            sentences = nltk.sent_tokenize(alignmentSent)
            if len(sentences) > 1:
                if parsePtr + 1 >= len(parses):
                    break
                parseWords += getParseWords(parses[parsePtr+1])
            alignmentWords = nltk.word_tokenize(alignmentSent)
            if alignmentWords[-1][-1] not in ('!', '?', '.', '"', "'", '`', ')'): #== ':':
                print(alignmentPtr, alignmentWords)
                alignmentWords += nltk.word_tokenize(getAlignmentSent(alignments[alignmentPtr+1]))

            match, disc, numMatched = getDiscourse(parseWords, alignmentWords, connectiveList)
            
            if not match:
                #handle weird corner cases
                if numMatched > 5:
                    m = None
                    for r in cornerCases:
                        m = r.match(alignmentSent)
                        if m is not None:
                            break
                          
                    if m is not None:
                        sentences = list(filter(None, m.groups()))
                        alignmentWords = reduce(operator.add,
                                                map(nltk.word_tokenize,
                                                    sentences))

                        #if there are 2 periods in the sentence, may not have sent_tokenized right
                        #i.e. it thinks "Name V. The" is a name so we shouldn't split
                        i = 0
                        parseWords = []
                        for i in range(len(sentences)):
                            if parsePtr + i >= len(parses):
                                break
                            currWords = getParseWords(parses[parsePtr+i])
                            sent = ' '.join(currWords)
                            m2 = r.match(sent)
                            if m2 is not None:
                                sentences = list(filter(None, m2.groups()))
                                parseWords += reduce(operator.add,
                                                     map(nltk.word_tokenize,
                                                         sentences))
                            else:
                                parseWords += currWords
                                
                            match, disc, numMatched = getDiscourse(parseWords, alignmentWords, connectiveList)
                            print(i, parseWords, alignmentWords, numMatched)
                            if match:
                                parsePtr += i
                                break
                print(parseWords, alignmentWords, match, disc, parsePtr, alignmentPtr, numMatched)
            if match:
                for d in disc:
                    with open(os.path.join(outputDir, d), 'a') as f:
                        f.write(alignments[alignmentPtr])
                        f.write("\n")
                parsePtr += 1
            #else:
            
            alignmentPtr += 1
        print(parsePtr, len(parses))
        assert(parsePtr == len(parses))
