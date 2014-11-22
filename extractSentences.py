
#if sys.version_info[0] < 3:
from __future__ import print_function

from parsedGigawordReader import ParsedGigawordReader
from sentenceReader import SentenceReader,extractAltlex

import sys
import math
from collections import defaultdict
import re

rwhitespace = re.compile("\s+")

def getConnectors():
    with open("/home/chidey/PDTB/Discourse/config/markers/markers") as f:
        return f.read().splitlines()

def getAltlexes():
    with open("/home/chidey/PDTB/Discourse/config/markers/altlexes") as f:
        return f.read().splitlines()

#read in causal and noncausal word pairs as dicts
#dont need numpy for this since we have sparse vectors for the sentence pairs
def getPairs(filename):
    scores = defaultdict(dict)

    with open(filename) as f:
        count = 0
        for line in f:
            count += 1
            word1,word2,score = line.split("\t")
            score = float(score)
            scores[word1][word2] = score
            if count % 100000 == 0:
                print ("done ", count)
            
    return scores

def getLength(scores):
    '''get the length of the vector'''
    total = 0
    for word1 in scores:
        for word2 in scores[word1]:
            total += scores[word1][word2]**2
            
    return math.sqrt(total)

def getCausalPairs():
    return getPairs("/home/chidey/PDTB/Discourse/working/wordpairs/gigaword2/content_word_pairs_stemmed_tsl/causal") #"/home/chidey/PDTB/Discourse/working/wordpairs/gigaword2/content_word_pairs_stemmed_tsl_tfidf/causal")

def getNonCausalPairs():
    return getPairs("/home/chidey/PDTB/Discourse/working/wordpairs/gigaword2/content_word_pairs_stemmed_tsl_tfidf/noncausal")

def getDotProduct(scores1, scores2):
    dp = 0
    for word1 in scores1:
        for word2 in scores1[word1]:
            if word1 in scores2 and word2 in scores2[word1]:
                dp += scores2[word1][word2] * scores1[word1][word2]

    return dp

if __name__ == '__main__':
    pgr = ParsedGigawordReader(sys.argv[1])
    altlexes = getAltlexes()
    causal = getCausalPairs()
    causalLength = getLength(causal)
    print("causal length is ", causalLength)
    '''
    nonCausal = getNonCausalPairs()
    nonCausalLength = getLength(nonCausal)
    '''

    altlexFile = open('newaltlexes', 'w')
    pairsFile = open('paired_bootstrapped_causal', 'w')

    seenSentences = set()

    try:
        for s in pgr.iterFiles():
            
            sr = SentenceReader(s)
            prevSentence = None

            for sentence in sr.iterSentences():
                lcSentence = ' '.join(sentence.words).lower()

                if lcSentence in seenSentences:
                    prevSentence = sentence
                    continue
                seenSentences.add(lcSentence)
                
                #only interested in sentences with no explicit discourse marker
                #because we already used these to create the scores
                #maybe add this later

                if prevSentence is None:
                    prevSentence = sentence
                    continue

                #if "that 's because" in lcSentence: #
                #    print(sentence.parse)
                #continue

                #only interested in sentences with no altlex
                if any(lcSentence.startswith(i) for i in altlexes):
                    prevSentence = sentence
                    continue

                #should also count how many of these sentences have some kind of clause as a DO
                
                #nonCausalCosSim = getDotProduct(instance, nonCausal)/(nonCausalLength*instanceLength)
                #print(instanceLength, causalCosSim)

                newAltlex = extractAltlex(sentence.parse)
                if newAltlex:
                    newAltlex = newAltlex.split()
                    #otherwise make dict out of cartesian product of words
                    instance = defaultdict(dict)
                    for word1 in prevSentence.stems:
                        for word2 in sentence.stems[len(newAltlex):]:
                            instance[word1][word2] = 1

                    instanceLength = getLength(instance)
                    causalCosSim = getDotProduct(instance, causal)/(causalLength*instanceLength)

                    print(causalCosSim, end="\t", file=altlexFile)

                    prevNamedEntity = ""
                    for i in range(len(newAltlex)):
                        if sentence.pos[i][0] == 'N' and sentence.ner[i] != "O":
                            if prevNamedEntity != sentence.ner[i]:
                                print(sentence.ner[i], end=" ", file=altlexFile)
                                print(sentence.ner[i], end=" ", file=pairsFile)
                            prevNamedEntity = sentence.ner[i]
                        elif sentence.pos[i].startswith('PR'):
                            print(sentence.pos[i], end=" ", file=altlexFile)
                            print(sentence.pos[i], end=" ", file=pairsFile)
                            prevNamedEntity = ""
                        elif sentence.pos[i][0].isalpha():
                            print(sentence.stems[i], end=" ", file=altlexFile)
                            print(sentence.stems[i], end=" ", file=pairsFile)
                            prevNamedEntity = ""
                    print(file=altlexFile)
                            
                    newAltlex = ' '.join(newAltlex).lower()
                    print("\t{}\t{}\t{}".format(' '.join(prevSentence.words).lower(), newAltlex, lcSentence.replace(newAltlex, '')), file=pairsFile)
                        
                    #find a clause (need to check for multiple levels)
                    #write these sentences out but only if they begin with a clause
                    #also write out the close

                prevSentence = sentence

    finally:
        altlexFile.close()
        pairsFile.close()
