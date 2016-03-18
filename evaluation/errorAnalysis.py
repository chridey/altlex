import collections

def makeTrueFalse(altlexes,labels,predictions):
        true_p = collections.defaultdict(int)
        false_p = collections.defaultdict(int)
        true_n = collections.defaultdict(int)
        false_n = collections.defaultdict(int)
        for i,p in enumerate(predictions):
            if p == labels[i]:
                if p:
                    true_p[altlexes[i]] += 1
                else:
                    true_n[altlexes[i]] += 1
            else:
                if p:
                    false_p[altlexes[i]] += 1
                else:
                    false_n[altlexes[i]] += 1
        return true_p,true_n,false_p,false_n

def printTotals(true_p, true_n, false_p, false_n):
    for i in (true_p, true_n, false_p, false_n):
        print(sum(i.values()), len(i))

def printTruePositives(true_p, true_n, false_p, false_n):
    for s in sorted(true_p, key=lambda x:true_p[x]):
        print('{} ***{}*** {} {} {}'.format(s, true_p[s], true_n[s], false_p[s], false_n[s]))

def printFalsePositives(true_p, true_n, false_p, false_n):
    for s in sorted(false_p, key=lambda x:false_p[x]):
	print('{} {} {} ***{}*** {}'.format(s, true_p[s], true_n[s], false_p[s], false_n[s]))
	
def printFalseNegatives(true_p, true_n, false_p, false_n):
    for s in sorted(false_n, key=lambda x:false_n[x]):
	print('{} {} {} {} ***{}***'.format(s, true_p[s], true_n[s], false_p[s], false_n[s]))	    
	
def printTrueNegatives(true_p, true_n, false_p, false_n):
    for s in sorted(true_n, key=lambda x:true_n[x]):
	print('{} {} ***{}*** {} {}'.format(s, true_p[s], true_n[s], false_p[s], false_n[s]))	    
	
