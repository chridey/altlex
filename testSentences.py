
#first extract sentences and their scores
#then get the rankings for the altlexes
#then determine how many of the altlexes are causal

from sentenceReader import SentenceRelationReader
from xmlReader import XMLReader
from extractSentences import CausalityScorer
from cosineSimilarityRanker import TaggedCosineSimilarityRanker

import sys

from collections import defaultdict
unique_tags = defaultdict(int)

#cs = CausalityScorer()
#csr = TaggedCosineSimilarityRanker()
xr = XMLReader(sys.argv[1])

for f in xr.iterFiles():
    #print(f)
    
    srr = SentenceRelationReader(f)
    prevSentence = None
    for sentence in srr.iterSentences():
        #print(sentence.parseString)
        #unique_tags[sentence.tag] +=1
        print(sentence.words, sentence.tag)
        continue
    
        ss = cs.scoreCausality(sentence, prevSentence)
 
        if ss is not None:
            csr.add(ss.causalCosSim, ss.modAltlex)

            #print(ss.newAltlex, ss.modAltlex, sentence.tag)

            csr.evaluate(ss.modAltlex, sentence.tag)
            
        prevSentence = sentence

exit()
for phrase,score in csr.iterByWeightedCosSim():
    print(phrase, csr.counts[phrase], csr.noncounts[phrase], csr.causal[phrase], csr.noncausal[phrase], score)
