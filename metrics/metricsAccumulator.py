import collections

import matplotlib.pyplot as plt

class MetricsAccumulator:
    '''accumulates metrics over time'''
    def __init__(self, numFolds, graphNames):
        self.numFolds = numFolds
        self.folds = list(range(numFolds)) + ['average']
        self.graphNames = graphNames
        
        self.f_measures = {i:collections.defaultdict(list) for i in self.folds}
        self.accuracies = {i:collections.defaultdict(list) for i in self.folds}
        self.precisions = {i:collections.defaultdict(list) for i in self.folds}
        self.recalls = {i:collections.defaultdict(list) for i in self.folds}

    def add(self,
            foldIndex,
            featureIndex,
            classifier,
            testing):
        
        accuracy = classifier.accuracy(testing)
        precision, recall = classifier.metrics(testing)
        f_measure = classifier.printResults(accuracy, precision, recall)
        
        self.accuracies[foldIndex][featureIndex].append(accuracy)
        self.precisions[foldIndex][featureIndex].append(precision)
        self.recalls[foldIndex][featureIndex].append(recall)
        self.f_measures[foldIndex][featureIndex].append(f_measure)

    def _average(self,
                 metric,
                 featureIndex,
                 iteration):
        result = [metric[i][featureIndex][iteration] for i in range(self.numFolds)]

        #print(result)
        return sum(result)/len(result)
            
    def average(self,
                featureIndex,
                iteration,
                classifier):


        accuracy = self._average(self.accuracies,
                                 featureIndex,
                                 iteration)
        precision = self._average(self.precisions,
                                  featureIndex,
                                  iteration)
        recall = self._average(self.recalls,
                               featureIndex,
                               iteration)
        self.accuracies['average'][featureIndex].append(accuracy)
        self.precisions['average'][featureIndex].append(precision)
        self.recalls['average'][featureIndex].append(recall)
        self.f_measures['average'][featureIndex].append(classifier.printResults(accuracy,
                                                                                precision,
                                                                                recall))
        
    def plotFmeasure(self):
        colors = ('r', 'b', 'y', 'g') #TODO
        plots = []
        for index,graphName in enumerate(self.graphNames):
            f_measures = self.f_measures['average'][graphName]
            plots.extend([range(len(f_measures)),
                          f_measures,
                          colors[index]])
                          

        plt.plot(*plots)
        plt.legend(self.graphNames)
        plt.axis(ymin=0, ymax=1) #precision can be at most 1 and not less than 0
        
    def savePlot(self, filename):
        plt.savefig(filename)
                          
        
    
