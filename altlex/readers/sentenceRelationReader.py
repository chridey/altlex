from chnlp.utils.readers.sentenceReader import Sentence, SentenceReader

class SentenceRelation(Sentence):
    def __init__(self, tag=None, **kwargs):
        #super().__init__(**kwargs)
        Sentence.__init__(self, **kwargs)
        self.tag = tag

class SentenceRelationReader(SentenceReader):
        def __init__(self, xmlroot):
            #super().__init__(xmlroot)
            SentenceReader.__init__(self, xmlroot)
            self.sentenceType = SentenceRelation

        def sentenceExtractor(self, sentence, parse=True):
            #kwargs = super().sentenceExtractor(sentence, parse)
            kwargs = SentenceReader.sentenceExtractor(self, sentence, parse)
            assert(sentence[-1].tag == 'tag')
            kwargs["tag"] = sentence[-1].text
            return kwargs
