import copy
import re

from nltk import Tree

class DataPoint:
    def __init__(self, dataDict):
        self._dataDict = dataDict
        self._altlexLower = None
        self._currParse = None

    @property
    def data(self):
        return self._dataDict
        
    def __hash__(self):
        return ' '.join(self.getPrevWords() + self.getCurrWords()).__hash__()

    def getTag(self):
        return self._dataDict['tag']

    @property
    def altlexLength(self):
        if 'altLexLength' in self._dataDict:
            self._dataDict['altlexLength'] = self._dataDict['altLexLength']
        return self._dataDict['altlexLength']

    @property
    def currSentenceLength(self):
        return len(self.getCurrWords())

    @property
    def prevSentenceLength(self):
        return len(self.getPrevWords())

    @property
    def currSentenceLengthPostAltlex(self):
        return len(self.getCurrWordsPostAltlex())

    def getSentences(self):
        #return both sentences in order as a string
        return ' '.join(self.getPrevWords() + self.getCurrWords())

    def getAltlexLemmasAndPos(self):
        if self.altlexLength > 0:
            return self.getAltlexLemmatized() + self.getAltlexPos()
        else:
            return []
        
    def getAltlex(self):
        if self.altlexLength > 0:
            return self.getCurrWords()[:self.altlexLength]
        else:
            return []

    def matchAltlex(self, phrase):
        a = self.getAltlex()
        if a is None:
            return False
        if a == phrase.split():
            return True        
        return False
    
    def getAltlexLemmatized(self):
        if self.altlexLength > 0:
            return [i.decode('utf-8') for i in self.getCurrLemmas()[:self.altlexLength]]
        else:
            return []

    def getAltlexStem(self):
        if self.altlexLength > 0:
            return self.getCurrStem()[:self.altlexLength]
        else:
            return []

    def getAltlexLower(self):
        if self._altlexLower is not None:
            return self._altlexLower

        self._altlexLower = [i.lower() for i in self.getCurrWords()[:self.altlexLength]]

        return self._altlexLower

    def getAltlexPos(self):
        if self.altlexLength > 0:
            return self._dataDict['sentences'][0]['pos'][:self.altlexLength]
        else:
            return []
    
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

    def getCurrWordsPostAltlex(self):
        return self.getCurrWords()[self.altlexLength:]

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

    def getCurrNer(self):
        return self._dataDict['sentences'][0]['ner']

    def getPrevNer(self):
        return self._dataDict['sentences'][1]['ner']

    def getCurrNerPostAltlex(self):
        return self._dataDict['sentences'][0]['ner'][self.altlexLength:]

    def getAltlexNer(self):
        return self._dataDict['sentences'][0]['ner'][:self.altlexLength]

    def getCurrParse(self):
        if self._currParse is not None:
            return self._currParse

        self.currParse = Tree.fromstring(self._dataDict['sentences'][0]['parse'])

        return self.currParse

    def getPrevDependencies(self):
        return self._dataDict['sentences'][1]['dependencies']
    
    def getCurrDependencies(self):
        return self._dataDict['sentences'][0]['dependencies']

    def getAltlexDependencies(self):
        return self._dataDict['altlex']['dependencies']

    def _getAltlex(self, form='words'):
        assert(form in ('words', 'lemmas', 'stems'))
        if self.altlexLength > 0:
            return self._dataDict['sentences'][0][form][:self.altlexLength]
        else:
            return []

    def _getCurr(self, form='words'):
        assert(form in ('words', 'lemmas', 'stems'))
        return self._dataDict['sentences'][0][form]

    def _getPrev(self, form='words'):
        assert(form in ('words', 'lemmas', 'stems'))
        return self._dataDict['sentences'][1][form]

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
