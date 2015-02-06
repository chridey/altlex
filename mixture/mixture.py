#data = [list(numpy.random.multivariate_normal([1,2,3], numpy.identity(3))) for i in range(100)]
#data += [list(numpy.random.multivariate_normal([13,6,11], 2*numpy.identity(3))) for i in range(200)]

#mixture modeling with constraints for binary classification
#only compatible with python 2.7 because of pymix

import numpy
from pymix import mixture

class ConstrainedMixture:
    def __init__(self, numDimensions):

        self.numDimensions = numDimensions
        
        f1 = mixture.MultiNormalDistribution(numDimensions, numpy.zeros(numDimensions), numpy.identity(numDimensions))
        f2 = mixture.MultiNormalDistribution(numDimensions, numpy.zeros(numDimensions), numpy.identity(numDimensions))

        #self.cm = mixture.ConstrainedMixtureModel(2, [0.5,0.5], [f1, f2])
        self.m = mixture.MixtureModel(2,[0.5,0.5], [f1, f2])
        
    def train(self, features, labels):
        assert(self.numDimensions == len(features[0]))

        print(len(features), len(features[0]))
        
        '''
        dataSet = mixture.ConstrainedDataSet()
        dataSet.fromList(features)

        pos_constr = numpy.zeros((len(features),len(features)), dtype='Float64')
        neg_constr = numpy.zeros((len(features),len(features)), dtype='Float64')

        #must be symmetric
        pos_indices = []
        neg_indices = []
        for index,label in enumerate(labels):
            if label == 1:
                pos_indices.append(index)
            elif label == 0:
                neg_indices.append(index)

        for i,index in enumerate(pos_indices):
            for index2 in pos_indices[i+1:]:
                pos_constr[index, index2] = 1.0
                pos_constr[index2, index] = 1.0

        for i,index in enumerate(neg_indices):
            for index2 in neg_indices[i+1:]:
                neg_constr[index, index2] = 1.0
                neg_constr[index2, index] = 1.0

        dataSet.setPairwiseConstraints(pos_constr,neg_constr)

        self.p = self.m.modelInitialization(dataSet, .5, .5, 3)
        print(self.p)
        self.m.EM(dataSet, 40, 0.1, len(labels)**2, len(labels)**2, self.p, 3)
        '''

        dataSet = mixture.DataSet()
        dataSet.fromList(features)
        dataSet.internalInit(self.m)
        
        self.m.modelInitialization(dataSet)
        self.m.EM(dataSet, 40, 0.1)
        
    def classify(self, features):
        return self.m.classify(features, .5, .5, self.p, 3)







