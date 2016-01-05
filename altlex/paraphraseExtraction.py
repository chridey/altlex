#1) a) iterate over the paired parsed data and extract potential altlexes (this should be done already)
#or b) load altlexes and parses from 1
#2) start with an initial training set using distant supervision to identify the causal datapoints
#at training time
#for k iterations:
#3) split each sentence on the correct altlex (for those without a known altlex use the alignment table to figure this out, i.e. does it overlap with the known altlexes, model/aligned.grow-diag-final (both ways?))
#4) calculate the KL divergence for every possible altlex but only causal/non-causal wordpairs (word pairs only for split sentences? no, not yet)
#5) do a matrix or tensor factorization on delta KLD weighted wordpairs
#6) extract any other features, lexical, binary, etc.
#6a) balance the data
#7) classify against unknown data, only add them to training if both pairs agree (the identified altlexes must be in each other's phrase tables)
#8) go to step 3

