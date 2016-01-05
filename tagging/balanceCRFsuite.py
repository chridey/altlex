import sys
from collections import defaultdict

from chnlp.utils.utils import balance

with open(sys.argv[1]) as f:
    t = f.read().split('\n\n')
#print(len(t))
finalData = defaultdict(list)
for s in t:
    if s.startswith('O'):
        finalData['O'].append(s)
    elif s.startswith('B-AL'):
        finalData['B-AL'].append(s)
    elif s.startswith('B-CAL'):
        finalData['B-CAL'].append(s)

#for i in finalData:
#    print (i, len(finalData[i]))
#exit()

#first oversample the 2nd most common class with the most common (probably altlex with not altlex)
#then oversample the next most (probably causal altlex with altlex)
#change labels to True/False and change them back
toBalance = sorted(finalData.items(), key=lambda x:len(x[1]), reverse=True)
#for i,j in toBalance:
    #pass
#    print(i, len(j))
#exit()

finalBalanced = [toBalance[0]]
while len(toBalance) > 1:
    label1,larger = toBalance.pop(0)
    label2,smaller = toBalance.pop(0)

    #print(label1)
    #print(type(larger), type(smaller))
    balanced = balance(zip(larger + smaller, [False]*len(larger) + [True]*len(smaller)))
    half = int(len(balanced)/2)
    newList = list(list(zip(*balanced[:half]))[0])
    finalBalanced.append((label2, newList))
    toBalance = [(label2, newList)] + toBalance

for i,j in finalBalanced:
    #print(i, len(j))
    #continue
    for k in j:
        #print(k)
        for l in k:
            if ord(l) < 128:
                print(l, end='')
        print()
