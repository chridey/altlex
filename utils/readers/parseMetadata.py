import operator

from altlex.utils import wordUtils
from altlex.utils import dependencyUtils
from altlex.featureExtraction.dataPoint import DataPoint

def getLemmas(lemmas):
    return reduce(operator.add, lemmas)

class ParseMetadata:
    def __init__(self, data):
        self.data = data

    @property
    def words(self):
        return getLemmas(self.data['words'])

    @property
    def wordsLower(self):
        return [i.lower().encode('utf-8') for i in self.words]

    @property
    def lemmas(self):
        return getLemmas(self.data['lemmas'])

    @property
    def lemmasLower(self):
        return [i.lower().encode('utf-8') for i in self.lemmas]

    @property
    def pos(self):
        return getLemmas(self.data['pos'])

    @property
    def ner(self):
        return getLemmas(self.data['ner'])

    @property
    def stems(self):
        return [wordUtils.snowballStemmer.stem(wordUtils.replaceNonAscii(i)) for i in self.lemmasLower]

    @property
    def dependencies(self):
        partialDependencies = []
        for depIndex,dep in enumerate(self.data['dep']):
            partialDependencies.append(dependencyUtils.tripleToList(dep, len(self.data['lemmas'][depIndex])))
        return dependencyUtils.combineDependencies(*partialDependencies)

    def datapoint(self, altlexStart, altlexEnd):
        newDependencies = dependencyUtils.splitDependencies(self.dependencies,
                                                            (altlexStart,
                                                             altlexEnd))
        dataPoint = {'altlexLength': altlexEnd-altlexStart,
                     'sentences': [{
                         'lemmas': self.lemmasLower[altlexStart:],
                         'words': self.words[altlexStart:], #changed from wordsLower
                         'stems': self.stems[altlexStart:],
                         'pos': self.pos[altlexStart:],
                         'ner': self.ner[altlexStart:],
                         'dependencies': newDependencies['curr']
                         },
                                   {
                                       'lemmas': self.lemmasLower[:altlexStart],
                                       'words': self.words[:altlexStart], #changed from wordsLower
                                       'stems': self.stems[:altlexStart],
                                       'pos': self.pos[:altlexStart],
                                       'ner': self.ner[:altlexStart],
                                       'dependencies': newDependencies['prev']
                                       }],
                     'altlex': {'dependencies': newDependencies['altlex']},
                     'dependencies': self.dependencies
                     }

        return DataPoint(dataPoint)



