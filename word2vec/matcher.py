class Matcher(object):
    def __init__(self, filename=None, phraseList=None):
        if filename:
            with open(filename) as f:
                phraseList = f.read().splitlines()
            
        for i in range(len(phraseList)):
            phraseList[i] = phraseList[i].replace("'s", " 's")
            phraseList[i] = phraseList[i].replace(":", " :")
            phraseList[i] = phraseList[i].replace(",", " ,")
            phraseList[i] = phraseList[i].replace('"', ' "')

        self.phraseList = phraseList

    def matches(self, sentence):
        mx = 0
        argmax = None

        for a in self.phraseList:
            alist = a.split()
            if sentence[:len(alist)] == alist:
                if len(alist) > mx:
                    mx = len(alist)
                    argmax = alist

        return argmax
