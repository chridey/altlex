from chnlp.utils.readers.sentenceReader import Sentence, SentenceReader

class SentenceRelation(Sentence):
    def __init__(self, tag=None, **kwargs):
        super().__init__(**kwargs)
        self.tag = tag

class SentenceRelationReader(SentenceReader):
        def __init__(self, xmlroot):
            self.root = xmlroot
            self.sentenceType = SentenceRelation

        def sentenceExtractor(self, sentence, parse=True):
            kwargs = super().sentenceExtractor(sentence, parse)
            assert(sentence[2].tag == 'tag')
            kwargs["tag"] = sentence[2].text
            return kwargs
