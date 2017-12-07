import sys
import json

from altlex.featureExtraction.neuralAltlexHandler import NeuralAltlexHandler

def load_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_file = sys.argv[1]
dev_file = sys.argv[2]
test_file = sys.argv[3]

print('loading train...')
train = load_data(train_file)

print('loading dev...')
dev = load_data(dev_file)

print('loading test...')
test = load_data(test_file)
            
handler = NeuralAltlexHandler(classifierFile=sys.argv[4], vocabFile=sys.argv[5])
print('building vocab...')
handler.build_vocab(train, min_count=0, lower=False)
handler.save_vocab()
print('training...')
handler.train(train, dev, test)
