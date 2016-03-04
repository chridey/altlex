from sklearn.metrics import precision_recall_fscore_support

def mostCommonClassEvaluator(newAltlexes, test):
    totals = [1.*sum(j.values()) for j in newAltlexes]
    print(totals)
    y_predict = []
    features,labels = zip(*test)
    for ix,f in enumerate(features):
        score,prediction = max(((newAltlexes[i][tuple(f['altlex'])]/totals[i],
                                 i) for i in range(len(newAltlexes))),
                               key=lambda x:x[0])
        y_predict.append(prediction)

        if prediction == 1:
            print(labels[ix],
                  f['altlex'],
                  newAltlexes[0][tuple(f['altlex'])]/totals[0],
                  newAltlexes[1][tuple(f['altlex'])]/totals[1])

    results = precision_recall_fscore_support(labels, y_predict)
    accuracy = 1-1.*sum(np.array(labels) ^ np.array(y_predict))/len(labels)
    return y_predict,accuracy,results
                                                        
def makeNgramOnlyFeatures(j):
    new_j = []
    for f,label in j:
        features = {}
        altlex_ngram = f['altlex'][:len(f['altlex'])/2]
        for i in range(len(altlex_ngram)-1):
            features['altlex_stem_' + altlex_ngram[i]] = 1
            features['altlex_stem_' + altlex_ngram[i] + '_' + altlex_ngram[i+1]] = 1
        features['altlex_stem_' + altlex_ngram[-1]] = 1
        new_j.append((features,label))

    return new_j
