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

    def _getAltlex(self, form='words'):
        assert(form in ('words', 'lemmas', 'stems'))
        if self.altlexLength > 0:
            return self._dataDict['sentences'][0][form][:self.altlexLength]
        else:
            return None

    def _getCurr(self, form='words'):
        assert(form in ('words', 'lemmas', 'stems'))
        return self._dataDict['sentences'][0][form]

    def _getPrev(self, form='words'):
        assert(form in ('words', 'lemmas', 'stems'))
        if self.altlexLength > 0:
            return self._dataDict['sentences'][1][form]
        else:
            return None

    def getStemsForPos(self, pos, part, form='stems'):
        if part == 'altlex':
            posList = self.getAltlexPos()
            stems = self._getAltlex(form)
        elif part == 'previous':
            posList = self.getPrevPos()
            stems = self._getPrev(form)
        elif part == 'current':
            posList = self.getCurrPos()
            stems = self._getCurr(form)
        else:
            raise NotImplementedError
        
        posInstances = []
        for (index,p) in enumerate(posList):
            if p.startswith(pos):
                posInstances.append(stems[index])
                break

        return posInstances
