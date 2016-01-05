from __future__ import print_function

import tempfile
import subprocess

from sklearn.base import BaseEstimator

from chnlp.ml.svm import SVM

class TransductiveSVMClassifier(BaseEstimator):
    kernels = {'linear': 0,
               'polynomial': 1,
               'rbf': 2,
               'tanh': 3}
    
    def __init__(self,
                 C=None,
                 positive_fraction=None,
                 kernel='rbf',
                 #gamma, etc
                 verbose=True
                 ):

        assert(kernel in TransductiveSVMClassifier.kernels)
        if positive_fraction:
            assert(0 <= positive_fraction <= 1)
        self.C = C
        self.positive_fraction = positive_fraction
        self.kernel = TransductiveSVMClassifier.kernels[kernel]
        self.verbose = verbose
        
    def get_params(self, deep=True):
        return {'C': self.C,
                'positive_fraction': self.positive_fraction,
                'kernel': self.kernel}

    def set_params(self, **params):
        if 'C' in params:
            self.C = params['C']
        if 'positive_fraction' in params:
            self.positive_fraction = params['positive_fraction']
        if 'kernel' in params:
            self.kernel = params['kernel']
        return self

    def _writeData(self, data, class_values):
        inputString = ''
        for i,dataPoint in enumerate(data):
            if class_values[i] == -1:
                label = 0
            elif class_values[i] == 0:
                label = -1
            else:
                label = 1
                
            inputString += '{} '.format(label)

            for j in range(len(dataPoint)):
                if dataPoint[j] != 0:
                    inputString +=  '{}:{} '.format(j, dataPoint[j])

            inputString += '\n'

        dataFile = tempfile.NamedTemporaryFile(mode='w')#, delete=False)
        print(inputString, file=dataFile, end='')

        return dataFile

    def fit(self, data, class_values):
        
        executableFilename = '/home/chidey/svm_light/svm_learn'

        dataFile = self._writeData(data, class_values)
        
        modelFile = tempfile.NamedTemporaryFile(mode='a')#, delete=False)

        #svm_learn -t <kernel> -c <C> -p <positive_fraction> <data> <model> 
        cmd = '{} -m 1000 -t {} '.format(executableFilename,
                                 self.kernel)
                        
        if self.C:
            cmd += '-c {} '.format(self.C)
        if self.positive_fraction:
            cmd += '-p {} '.format(self.positive_fraction)

        #set gamma to be 1/n_features
        if TransductiveSVMClassifier.kernels['rbf'] == self.kernel:
            cmd += '-g {} '.format(.0016) #(1/len(data[0]))

        cmd += '{} {}'.format(dataFile.name,
                              modelFile.name)

        print(cmd)
              
        if self.verbose:
            p = subprocess.Popen(cmd.split())
        else: #suppress stdout and stderr
            p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #try:
        p.communicate()

        dataFile.close()

        #we need the model file to hang around, so this relies on the tempfile
        #being implicitly closed later when garbage collected
        self.modelFile = modelFile

        return self

    def _classify(self, data, class_values, deleteOutput=True):
        executableFilename = '/home/chidey/svm_light/svm_classify'
        
        dataFile = self._writeData(data, class_values)

        outputFile = tempfile.NamedTemporaryFile(mode='a+')#, delete=False)
        
        #svm_classify example_file model_file output_file
        cmd = '{} {} {} {}'.format(executableFilename,
                                 dataFile.name,
                                 self.modelFile.name,
                                 outputFile.name)

        #capture stdout and stderr
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #try:
        (out, err) = p.communicate()

        dataFile.close()

        return outputFile, out

    def score(self, data, class_values, deleteOutput=True):

        outputFile, out = self._classify(data, class_values)
        
        #looking for line, for example
        #Accuracy on test set: 96.00% (576 correct, 24 incorrect, 600 total)
        searchString = b'Accuracy on test set: '
        i = out.find(searchString) + len(searchString)
        score = float(out[i:].split()[0][:-1])
        
        outputFile.close()

        return score
        
    def predict(self, data):
        outputFile, out = self._classify(data, [-1 for i in data])

        outputFile.seek(0)
        outputLines = outputFile.read().splitlines()
        
        #read the output file and assign labels
        labels = []
        for line in outputLines:
            label = float(line)
            labels.append(label > 0)
            
        outputFile.close()

        return labels
    
class TransductiveSVM(SVM):
    def __init__(self):
        super().__init__()
        self.classifier = TransductiveSVMClassifier(C=10,
                                                    positive_fraction=.2)

    def metrics(self, testing, transform=True):
        truepos = 0
        trueneg = 0
        falsepos = 0
        falseneg = 0

        features, labels = zip(*testing)
        if transform:
            X = self._transform(features)
        else:
            X = features

        assigned = self.classifier.predict(X)
              
        for i,label in enumerate(labels):
            if assigned[i] == label:
                if label == True:
                    truepos += 1
                else:
                    trueneg += 1
            elif label == False:
                falsepos +=1
            else:
                falseneg += 1

        print(truepos, trueneg, falsepos, falseneg)

        precision, recall = self._calcPrecisionAndRecall(truepos, trueneg, falsepos, falseneg)
        
        return precision, recall
            
        
    
    
