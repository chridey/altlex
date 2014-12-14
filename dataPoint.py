from nltk import Tree

class DataPoint:
    def __init__(self, dataDict):
        self._dataDict = dataDict
        self._altlexLower = None
        self._currParse = None
        
    def __hash__(self):
        return ' '.join(self.getPrevWords() + self.getCurrWords()).__hash__()

    @property
    def altlexLength(self):
        return self._dataDict['altlexLength']
        
    def getAltlex(self):
        if self.altlexLength > 0:
            return self.getCurrWords()[:self.altlexLength]
        else:
            return None

    def getAltlexLemmatized(self):
        if self.altlexLength > 0:
            return self.getCurrLemmas()[:self.altlexLength]
        else:
            return None

    def getAltlexStem(self):
        if self.altlexLength > 0:
            return self.getCurrStem()[:self.altlexLength]
        else:
            return None

    def getAltlexLower(self):
        if self._altlexLower is not None:
            return self._altlexLower

        self._altlexLower = [i.lower() for i in self.getCurrWords()[:self.altlexLength]]

        return self._altlexLower

    def getAltlexPos(self):
        if self.altlexLength > 0:
            return self._dataDict['sentences'][0]['pos'][:self.altlexLength]
        else:
            return None
    
    def getCurrLemmas(self):
        return self._dataDict['sentences'][0]['lemmas']

    def getPrevLemmas(self):
        return self._dataDict['sentences'][1]['lemmas']

    def getCurrStem(self):
        return self._dataDict['sentences'][0]['stems']

    def getPrevStem(self):
        return self._dataDict['sentences'][1]['stems']

    def getCurrWords(self):
        return self._dataDict['sentences'][0]['words']

    def getPrevWords(self):
        return self._dataDict['sentences'][1]['words']

    def getCurrPos(self):
        return self._dataDict['sentences'][0]['pos']

    def getPrevPos(self):
        return self._dataDict['sentences'][1]['pos']

    def getCurrLemmasPostAltlex(self):
        return self.getCurrLemmas()[self.altlexLength:]

    def getCurrStemPostAltlex(self):
        return self.getCurrStem()[self.altlexLength:]

    def getCurrPosPostAltlex(self):
        return self.getCurrPos()[self.altlexLength:]

    def getCurrParse(self):
        if self._currParse is not None:
            return self._currParse

        self.currParse = Tree.fromstring(self._dataDict['sentences'][0]['parse'])

        return self.currParse

    def getStemsForPos(self, pos, part):
        if part == 'altlex':
            posList = self.getAltlexPos()
            stems = self.getAltlexStem()
        elif part == 'previous':
            posList = self.getPrevPos()
            stems = self.getPrevStem()
        elif part == 'current':
            posList = self.getCurrPos()
            stems = self.getCurrStem()
        else:
            raise NotImplementedError
        
        posInstances = []
        for (index,p) in enumerate(posList):
            if p.startswith(pos):
                posInstances.append(stems[index])
                break

        return posInstances
