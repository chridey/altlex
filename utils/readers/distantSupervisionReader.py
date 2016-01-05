import re
import xml.etree.ElementTree as ET
from copy import deepcopy

from chnlp.altlex.readers.sentenceRelationReader import SentenceRelation,SentenceRelationReader

from chnlp.config import settings

class CausalTwoClauseSentence:
    def __init__(self, *args, **kwargs):
        self.previous = SentenceRelation(*args, **kwargs)
        self.current = deepcopy(self.previous)
        self.tag = self.previous.tag
        
        if self.tag == 'causal':
            index = self.previous.words.index('because')
        else:
            index = self.previous.words.index('but')
        '''
        self.connective,self.index = self.previous.tag.split(',')
        self.index = int(self.index)
        index = self.previous.words.index(self.connective)
        '''
        for attr in 'words', 'lemmas', 'pos', 'ner':
            if getattr(self.previous, attr, None) is not None:
                setattr(self.previous, attr, getattr(self.previous, attr)[:index])
                
            if getattr(self.current, attr, None) is not None:
                setattr(self.current, attr, getattr(self.current, attr)[index+1:])
            
    @property
    def valid(self):
        return True

class DistantSupervisionReader(SentenceRelationReader):
    def __init__(self, document, markers=None, causalMarkers=None, noncausalMarkers=None):
        self.document = "".join(document)
        self.sentenceType = CausalTwoClauseSentence

        if markers is None:
            with open(settings.markerFile) as f:
                markers = f.read().splitlines()

        self.markers = {}
        for marker in markers:
            self.markers[marker] = len(self.markers)
            
        because = "<word>because</word>"
        so = "<word>so</word>"
        if causalMarkers is None:
            self.causal_connectives = [because] #, so]
        else:
            self.causal_connectives = causalMarkers
            
        but = "<word>but</word>"
        when = "<word>when</word>"
        also = "<word>also</word>"
        whil = "<word>while</word>"
        iff = "<word>if</word>"
        though = "<word>though</word>"
        however = "<word>however</word>"
        if noncausalMarkers is None:
            self.not_causal_connectives = [but] #, though] #, whil, iff, when]
        else:
            self.not_causal_connectives = noncausalMarkers

    def _findConnective(self, text):
        tag = None
        for connective in self.causal_connectives:
            if connective in text:
                tag = "<tag>causal</tag>"
                break
        if tag is None:
            for connective in self.not_causal_connectives:
                if connective in text:
                    tag = "<tag>notcausal</tag>"
                    break
        return tag
            
    def iterSentences(self, parse=True):
        
        for match in re.finditer(r"(<sentence id=\"\d+\">.*?)(</sentence>)",
                                 self.document, flags=re.DOTALL):
            tag = None
            text = match.group(1)
            end =  match.group(2)
            try:
                pass #print(text)
            except UnicodeEncodeError:
                pass

            tag = self._findConnective(text)
            
            if tag is not None:
                #TODO, possibly: make sure that these are discourse connectives for two clauses
                try:
                    xmlParse = ET.fromstring(text + "\n" + tag + "\n" + end)
                except UnicodeEncodeError:
                    continue
                sentence = self.sentenceType(**self.sentenceExtractor(xmlParse, parse))
                if sentence.valid:
                    yield sentence


class DistantSupervisionDiscourseReader(DistantSupervisionReader):
    def __init__(self, *args, **kwargs):
        DistantSupervisionReader.__init__(self, *args, **kwargs)
        self.sentenceType = TwoClauseSentence
        
    def _findConnective(self, text):
        tag = None
        for connective in self.markers:
            if "<word>" + connective + "</word>" in text:
                tag = "<tag>{},{}</tag>".format(connective, self.markers[connective])
                break
        return tag

class TwoClauseSentence:
    def __init__(self, *args, **kwargs):
        self.previous = SentenceRelation(*args, **kwargs)
        self.current = deepcopy(self.previous)

        self.connective,self.index = self.previous.tag.split(',')
        self.index = int(self.index)
        index = self.previous.words.index(self.connective)

        for attr in 'words', 'lemmas', 'pos', 'ner':
            if getattr(self.previous, attr, None) is not None:
                setattr(self.previous, attr, getattr(self.previous, attr)[:index])
                
            if getattr(self.current, attr, None) is not None:
                setattr(self.current, attr, getattr(self.current, attr)[index+1:])
            
    @property
    def valid(self):
        return True
