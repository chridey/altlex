from chnlp.utils.readers.xmlReader import XMLReader
from chnlp.utils.readers.sentenceReader import SentenceReader

class SemaforXMLReader(XMLReader):
    def __init__(self, xml_file_name):
        self.file_name = xml_file_name
        self.root = 'corpus'

    def iterFiles(self):
        for documents in super().iterFiles():
            assert(documents.tag == 'documents')
            for document in documents:
                assert(document.tag == 'document')                
                for paragraphs in document:
                    assert(paragraphs.tag == 'paragraphs')                
                    for paragraph in paragraphs:
                        assert(paragraph.tag == 'paragraph')                
                        yield paragraph

class FrameSentence:
    def __init__(self, text, frames):
        self.text = text
        self.frames = frames

class FrameSentenceReader(SentenceReader):
    def __init__(self, xmlRoot):
        self.root = xmlRoot
        self.sentenceType = FrameSentence
        self.document = 'paragraph'

    def sentenceExtractor(self, sentence, parse=False):
        assert(sentence.tag == 'sentence')
        text = sentence[0]
        assert(text.tag == 'text')
        text = text.text
        
        frames = set()
        annotationSets = sentence[1]
        assert(annotationSets.tag == 'annotationSets')
        for annotationSet in annotationSets:
            assert(annotationSet.tag == 'annotationSet')            
            #extract attribute
            frameName = annotationSet.attrib['frameName']
            #TODO: extract layers
            frames.add(frameName)

        return {'text': text,
                'frames': frames}

class FrameNetManager:
    def __init__(self, filename):
        self._lookup = {}
        
        ser = SemaforXMLReader(filename)
        for doc in ser.iterFiles():
            fr = FrameSentenceReader(doc)
            for fs in fr.iterSentences():
                self._lookup[''.join(fs.text.split())] = fs.frames

    def getFrames(self, text):
        return self._lookup[text]

    @staticmethod
    def isCausalFrame(frame):
        frame = frame.lower()
        if 'caus' in frame or frame in ('reason', 'explanation', 'effect', 'trigger') or 'purpose' in frame or 'required' in frame or 'consequence' in frame or 'result' in frame or 'response' in frame or 'enabled' in frame:
            return True
        else:
            return False

    @staticmethod
    def isAntiCausalFrame(frame):
        if frame in {
            'Requirements',
            'Have_as_requirement',
            'Statement',
            'Occasion',
            'Supplier',
            'Thriving',
            'Seeking',
            'Expectation',
            'Being_necessary',
            'Clarity_of_resolution',
            'Circumstances',
            'Relative_time',
            'Idiosyncrasy',
            'Compliance',
            'Aggregate',
            'Estimation',
            'Bringing',
            'Perception_active',
            'Responding_entity',
            'Evaluative_comparison',
            'Justifying',
            'Waiting',
            'Taking_sides',
            'Questioning',
            'Proposed_action',
            'Evaluee',
            'Descriptor',
            'Evidence',
            'Defend',
            'Activity_ongoing',
            'Boundary',
            'Coming_up_with',
            'Interval',
            'Explaining_the_facts',
            'Communicate_categorization',
            'Terms_of_agreement',
            'Requirement',
            'Process',
            'State',
            'Remembering_experience',
            'Defender',
            'Categorization',
            'Indicating',
            'Appointing',
            'Consideration',
            'Depictive',
            'Coming_to_believe',
            'Proposition',
            }:
            return True
        else:
            return False
        
if __name__ == '__main__':
    import sys

    ser = SemaforXMLReader(sys.argv[1])
    for doc in ser.iterFiles():
        fr = FrameSentenceReader(doc)
        for fs in fr.iterSentences():
            print(fs.text, fs.frames)
