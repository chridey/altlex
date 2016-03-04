import gzip
import json

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
            self.prevDependencies = dataPoint.getPrevDependencies(),
            self.currDependencies = dataPoint.getCurrDependencies(),
            self.altlexDependencies = dataPoint.getAltlexDependencies()
            self.features = features
            self.label = label
            self.datumId = datumId
            self.sentenceId = sentenceId
        else:
            for param in params:
                setattr(self, param, params[param])
            
    @property
    def JSON(self):
        return {'altlex': self.altlex,
                'prevDependencies': self.prevDependencies,
                'currDependencies': self.currDependencies,
                'altlexDependencies': self.altlexDependencies,
                'features': self.features,
                'label': self.label,
                'datumId': self.datumId,
                'sentenceId': self.sentenceId}

    @classmethod
    def fromJSON(cls, data):
        return cls(None, None, None, None, params=data)
    
class DataPointMetadataList(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)

    def save(self, filename):
        with gzip.open(filename, 'w') as f:
            json.dump(self.JSON, f)

    def load(self, filename):
        with gzip.open(filename) as f:
            return self.fromJSON(json.load(f))

    @property
    def JSON(self):
        return [i.JSON for i in self]

    @classmethod
    def fromJSON(cls, data):
        return cls([DataPointMetadata.fromJSON(i) for i in data])
                 
    #TODO
    #transform with feature vectorizer
    #get subset of features
    #get datum indices
           
