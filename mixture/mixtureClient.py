#python3 client to allow using python2 mixture module 

import socket
import json

from sklearn.base import BaseEstimator

from chnlp.ml.sklearner import Sklearner

class MixtureClientClassifier(BaseEstimator):
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port

    #need to define fit, score, and predict
    def fit(self, data, class_values):
        dataDict = {'data': data,
                    'class_values': class_values}
        package = json.dumps(dataDict)
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientSocket.connect((self.host, self.port))
        clientSocket.sendall((package + "\n").encode('utf-8'))

        received = clientSocket.recv(1024)

        clientSocket.close()

        return self

    def predict(self, data):
        dataDict = {'data': data}
        package = json.dumps(dataDict)
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientSocket.connect((self.host, self.port))
        clientSocket.sendall((package + "\n").encode('utf-8'))

        buf = bytes()
        while True:
            buf += clientSocket.recv(1024)
            if not buf or buf[-1] == '\n':
                break

        import sys
        print('buffer!!!')
        print(buf.decode('utf-8'))
        sys.stdout.flush()
            
        labels = json.loads(buf.decode('utf-8'))

        clientSocket.close()

        return labels

    def score(self, data, class_values, deleteOutput=True):
        
        labels = self.predict(data)
        numCorrect = 0
        for index, label in enumerate(labels):
            if class_values[index] == label:
                numCorrect += 1
        return numCorrect/len(labels)

class MixtureClient(Sklearner):
    def __init__(self):
        super().__init__()
        self.classifier = MixtureClientClassifier()

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

    def close(self):
        self.classifier.close()
        
    
    

