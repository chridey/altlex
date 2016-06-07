import gzip
import json
import csv
import collections

from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np

from altlex.utils import wordUtils

def formatDependencies(words,deps):
    datum = [('ROOT', None, None)]
    for i,j in enumerate(deps):
        if j is None:
            datum.append((None,None,None))
        else:
            datum.append((words[i].lower(), j[0], j[1]+1))
    return datum

def modifyDependencies(dataPoint, label):
    prevDependencies = dataPoint.getPrevDependencies()
    currDependencies = dataPoint.getCurrDependencies()
    prevWords = dataPoint.getPrevWords()
    currWords = dataPoint.getCurrWordsPostAltlex()
    
    data = []
    for words,deps in ((prevWords, prevDependencies),
                       (currWords, currDependencies)):

        datum = [('ROOT', None, None)]
        for i,j in enumerate(deps):
            if j is None:
                datum.append((None,None,None))
            else:
                datum.append((words[i].lower(), j[0], j[1]+1))

        data.append(datum)
        
    return data[0],data[1],label

class DataPointMetadata:
    def __init__(self,
                 dataPoint,
                 features,
                 label,
                 datumId=None,
                 sentenceId=None,
                 params=None):

        if params is None:
            self.altlex = dataPoint.getAltlexLemmasAndPos()
            self.words = [dataPoint.getPrevWords(),
                          dataPoint.getAltlex(),
                          dataPoint.getCurrWordsPostAltlex()]
            self.dependencies = modifyDependencies(dataPoint, label)
            self.origDependencies = dataPoint._dataDict.get('dependencies', None)
            self.features = features
            self.label = label
            self.datumId = datumId
            self.sentenceId = sentenceId
        else:
            for param in params:
                setattr(self, param, params[param])

    @property
    def sentence(self):
        ret = []
        for i in self.words:
            ret += i
        return ret
            
    @property
    def JSON(self):
        return {'altlex': self.altlex,
                'words': getattr(self, 'words', None),
                'dependencies': self.dependencies,
                'origDependencies': getattr(self, 'origDependencies', None),
                'features': self.features,
                'label': self.label,
                'datumId': self.datumId,
                'sentenceId': self.sentenceId}

    @classmethod
    def fromJSON(cls, data):
        return cls(None, None, None, None, params=data)

    @property
    def CSV(self):
        return self.sentenceId,self.datumId,(' '.join(self.words[0])).encode('utf-8'),(' '.join(self.words[1])).encode('utf-8'),(' '.join(self.words[2])).encode('utf-8')

    @property
    def testCSV(self):
        return (' '.join(self.words[0])).encode('utf-8'),(' '.join(self.words[1])).encode('utf-8'),(' '.join(self.words[2])).encode('utf-8'),self.label

class DataPointMetadataList(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)

    def save(self, filename):
        with gzip.open(filename, 'w') as f:
            json.dump(self.JSON, f)

    @classmethod
    def load(cls, filename):
        with gzip.open(filename) as f:
            return DataPointMetadataList.fromJSON(json.load(f))

    def dedupe(self, dataPointMetadataList, verbose=False):
        ret = []
        dedupe = set()

        print(len(self), len(dataPointMetadataList))
        for i in dataPointMetadataList:
            dedupe.add(' '.join([j.lower() for j in i.sentence]))

        for i in self:
            if ' '.join([j.lower() for j in i.sentence]) not in dedupe:
                ret.append(i)

        if verbose:
            print('Found {} duplicates'.format(len(self) - len(ret)))
            
        return DataPointMetadataList(ret)

    def dedupeIndices(self, dataPointMetadataList, verbose=False):
        ret = []
        dedupe = dataPointMetadataList.datumIndices

        print(len(self), len(dataPointMetadataList))
        for i in self:
            if i.datumId not in dataPointMetadataList:
                ret.append(i)

        if verbose:
            print('Found {} duplicates'.format(len(self) - len(ret)))
            
        return DataPointMetadataList(ret)

    @property
    def JSON(self):
        return [i.JSON for i in self]

    @classmethod
    def fromJSON(cls, data):
        return cls([DataPointMetadata.fromJSON(i) for i in data])

    def CSV(self, csvfile):
        fieldnames = ['sentenceId', 'datumId', 'prevWords', 'altlex', 'currWords']
        with open(csvfile, 'wb') as f:
            csvwriter = csv.writer(f,
                                   delimiter='\t')
            csvwriter.writerow(fieldnames)
            for i in self:
                csvwriter.writerow(i.CSV)

    def testCSV(self, csvfile):
        fieldnames = ['prevWords', 'altlex', 'currWords', 'label']
        with open(csvfile, 'wb') as f:
            csvwriter = csv.writer(f,
                                   delimiter='\t')
            csvwriter.writerow(fieldnames)
            for i in self:
                csvwriter.writerow(i.testCSV)

    def matchWithCSV(self, csvfile, labelLookup):
        lookup = {}
        for index,i in enumerate(self):
            _,_,prev,altlex,curr = i.CSV
            lookup[(prev,altlex,curr)] = index

        ret = []
        total = 0
        dupes = set()
        with open(csvfile, 'rbU') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                prev = row['prevwords']
                altlex = row['altlex']
                curr = row['currwords']

                if (prev,altlex,curr) in dupes:
                    continue
                dupes.add((prev,altlex,curr))
                
                tag = row['indicate_whether_the_underlined_phrase_indicates__a_causal_relationship_between_the_words_immediately_before_and_after__if_the_relationship_is_causal_indicate_the_direction_of_the_relationship_if_a_causes_b_mark_the_example_as_result_and_if_b_causes_a_mark_the_example_as_reason']
                index = lookup.get((prev,altlex,curr), None)

                if index is None:
                    print('problem with {} {} {}'.format(prev, altlex, curr))
                    total += 1
                    continue
                datum = DataPointMetadata.fromJSON(self[index].JSON)
                datum.label = labelLookup[tag]
                ret.append(datum)

        print(total)
        return DataPointMetadataList(ret)
                
    @property
    def altlexes(self):
        altlexes = collections.defaultdict(collections.Counter)
        for i in self:
            altlexes[i.label][tuple(i.altlex)] += 1
        return altlexes

    @property
    def combinedAltlexes(self):
        altlexes = collections.defaultdict(collections.Counter)
        for index,label in enumerate(self.iterLabels(True)):
            altlexes[label][tuple(self[index].altlex)] += 1
        return altlexes

    @property
    def causalAltlexes(self):
        causalAltlexes = set()
        for i in self.altlexes:
            if i != 0:
                causalAltlexes.update(self.altlexes[i].keys())
        return causalAltlexes
    
    @property
    def datumIndices(self):
        return {i.datumId for i in self}

    def sample(self, numEach, predictions, combined=False):
        subsets = [[], [], []]
        for index,i in enumerate(self):
            if len(i.words[0]) > 3 and len(i.words[2]):
                subsets[predictions[index]].append(i)

        print([len(i) for i in subsets])
        ret = []
        for index,subset in enumerate(subsets):
            if combined and index == 1:
                num = numEach * 2
            else:
                num = numEach
            if len(subset):
                ret.extend(np.random.choice(subset, min(num,len(subset)), False).tolist())
            
        return DataPointMetadataList(ret)

    def split(self, percentage, stratified=True): #TODO: not stratified
        datumIds,labels = zip(*((self[i].datumId,self[i].label) for i in range(0,len(self),2)))
        sss = StratifiedShuffleSplit(labels, 1, test_size=percentage, random_state=0)

        trainIndices,testIndices = list(sss)[0]
        datumIds = np.array(datumIds)

        return set(datumIds[trainIndices]), set(datumIds[testIndices])

    def subsets(self, *indicesList):
        ret = [[] for i in range(len(indicesList))]
        for i in self:
            for index,indices in enumerate(indicesList):
                if i.datumId in indices:
                    ret[index].append(i)
                    break
        return [DataPointMetadataList(i) for i in ret]

    def iterLabels(self, combined=False):
        for i in self:
            if combined:
                yield i.label if i.label == 0 else 1
            else:
                yield i.label

    def updateLabels(self, labels):
        for index,i in enumerate(self):
            i.label = labels[index]

    def iterFeatures(self):
        for i in self:
            yield i.features

    #TODO: this doesnt really belong here
    def withConnectiveOnly(self, labelLookup):
        definiteReason = wordUtils.reason_markers
        possibleReason = definiteReason|wordUtils.possible_reason_markers
        definiteResult = wordUtils.result_markers - {('so', 'IN')}
        possibleResult = wordUtils.result_markers|wordUtils.possible_result_markers
        definiteNon = wordUtils.noncausal_markers
        possibleNon = wordUtils.noncausal_markers|wordUtils.possible_noncausal_markers

        ret = []
        for i in range(0, len(self), 2):
            first = tuple(self[i].altlex[:len(self[i].altlex)//2])
            second = tuple(self[i+1].altlex[:len(self[i+1].altlex)//2])

            if (first in definiteReason and second in possibleReason) or (second in definiteReason and first in possibleReason):
                metaLabel = 'reason'
            elif (first in definiteResult and second in possibleResult) or (second in definiteResult and first in possibleResult):
                metaLabel = 'result'
            elif (first in definiteNon and second in possibleNon) or (second in definiteNon and first in possibleNon):
                metaLabel = 'notcausal'
            else:
                continue

            for j in ((i, i+1)):
                data = self[j]
                data.label = labelLookup[metaLabel]
                ret.append(data)

        return DataPointMetadataList(ret)

    def withConnectiveOneSide(self, labelLookup):
        ret = []
        for i in range(0, len(self), 2):
            first_lemmas = tuple(self[i].altlex[:len(self[i].altlex)//2])
            first_pos = tuple(self[i].altlex[len(self[i].altlex)//2:])
            second_lemmas = tuple(self[i+1].altlex[:len(self[i+1].altlex)//2])
            second_pos = tuple(self[i+1].altlex[len(self[i+1].altlex)//2:])

            #allow preposition at beginning or end
            if any(lemmas in wordUtils.reason_markers or (lemmas[:-1] in wordUtils.reason_markers and pos[-1] in ('TO', 'IN')) or (lemmas[1:] in wordUtils.reason_markers and pos[0] in ('TO', 'IN')) for lemmas,pos in ((first_lemmas,first_pos),(second_lemmas,second_pos))):
                metaLabel = 'reason'
            elif any(lemmas in wordUtils.result_markers or lemmas+pos in wordUtils.result_markers or (lemmas[:-1] in wordUtils.result_markers and pos[-1] in ('TO', 'IN')) or (lemmas[1:] in wordUtils.result_markers and pos[0] in ('TO', 'IN')) for lemmas,pos in ((first_lemmas,first_pos),(second_lemmas,second_pos))):
                metaLabel = 'result'
            elif any(lemmas in wordUtils.noncausal_markers or (lemmas[:-1] in wordUtils.noncausal_markers and pos[-1] in ('TO', 'IN')) or (lemmas[1:] in wordUtils.noncausal_markers and pos[0] in ('TO', 'IN')) for lemmas,pos in ((first_lemmas,first_pos),(second_lemmas,second_pos))):
                metaLabel = 'notcausal'
            else:
                continue

            for j in ((i, i+1)):
                data = self[j]
                data.label = labelLookup[metaLabel]
                ret.append(data)

        return DataPointMetadataList(ret)
        
    
    #TODO
    #transform with feature vectorizer
    #get subset of features
    #get datum indices
           
