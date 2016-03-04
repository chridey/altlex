def makeTrueFalse(features,labels,predictions):
        true_p = collections.defaultdict(int)
        false_p = collections.defaultdict(int)
        true_n = collections.defaultdict(int)
        false_n = collections.defaultdict(int)
        for i,p in enumerate(predictions):
            if p == labels[i]:
                if p:
                    true_p[tuple(features[i]['altlex'])] += 1
                else:
                    true_n[tuple(features[i]['altlex'])] += 1
            else:
                if p:
                    false_p[tuple(features[i]['altlex'])] += 1
                else:
                    false_n[tuple(features[i]['altlex'])] += 1
        return true_p,true_n,false_p,false_n

def printTotals(true_p, true_n, false_p, false_n):
    for i in (true_p, true_n, false_p, false_n):
        print(sum(i.values()), len(i))

def printTruePositives(true_p, true_n):
    for s in sorted(true_p, key=lambda x:true_p[x]):
        print(s, true_p[s], true_n[s])

def printFalsePositives(false_p, true_p, true_n):
    for s in sorted(false_p, key=lambda x:false_p[x]):
        print(s, false_p[s], true_p[s], true_n[s])

def printFalseNegatives(false_n, true_p, true_n):
    for s in sorted(false_n, key=lambda x:false_n[x]):
        print(s, false_n[s], true_p[s], true_n[s])

def printTrueNegatives(true_n, false_n):
    for s in sorted(true_n, key=lambda x:true_n[x]):
        print(s, true_n[s], false_n[s])
