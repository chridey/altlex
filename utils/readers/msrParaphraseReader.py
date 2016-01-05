import nltk

class MsrParaphraseReader:
    def __init__(self, filename):
        self.filename = filename

    def iterSentences(self):
        with open(self.filename) as f:
            for line in f:
                if not line or line.startswith('Quality'):
                    continue
                
                label, id1, id2, sent1, sent2 = line.strip().split('\t')
                yield label, sent1, sent2

    def asTrainingSet(self):
        X = []
        y = []
        for label,sent1,sent2 in self.iterSentences():
            X.extend([[i.lower() for i in nltk.word_tokenize(sent1)],
                       [i.lower() for i in nltk.word_tokenize(sent2)]])
            y.append(label)
        return X,y
