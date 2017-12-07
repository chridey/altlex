import json
import collections
import os

import numpy as np

from altlex.featureExtraction.altlexHandler import AltlexHandler
from altlex.neural.altlexLSTMClassifier import train as train_lstm, features, init_model, load, predict
from altlex.featureExtraction.dataPoint import findAltlexes

derived_features = frozenset(('deps', 'altlex_position', 'root_position'))
MAX_LEN = 30

class NeuralAltlexHandler(AltlexHandler):
    def __init__(self,    
                 classifierFile=None,
                 vocabFile=None,
                 altlexFile=None,
                 features=features,
                 verbose=False):

        self._classifierFile = classifierFile
        self._classifier = None #best classifier trained on wikipedia, load from settings
        self._vocabFile = vocabFile
        self._vocab = None
        
        self.altlexFile = altlexFile

        self._features = features
        
        self._causalAltlexes = None #causal altlexes, load from settings
        self._nonCausalAltlexes = None

    @property
    def classifierFile(self):
        if self._classifierFile is None:
            self._classifierFile = os.path.join(os.path.dirname(__file__), 'model.npz')                                                            
        return self._classifierFile
    
    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = init_model(self.vocab)
            load(self._classifier, self.classifierFile)
        return self._classifier

    @property
    def vocabFile(self):
        if self._vocabFile is None:
            self._vocabFile = os.path.join(os.path.dirname(__file__), 'vocab.json')
        return self._vocabFile
    
    @property
    def vocab(self):
        if self._vocab is None:
            with open(self.vocabFile) as f:
                self._vocab = json.load(f)
        return self._vocab

    def build_vocab(self, sentences, min_count=5, lower=True):
        counts = collections.defaultdict(collections.Counter)
        
        for sentence in sentences:
            for feature in self._features:
                processed = self.get_features(sentence, feature)
                for item in processed:
                    if lower:
                        item = item.lower()

                    counts[feature][item] += 1
                    
        self._vocab = {feature:{'UNK': 0} for feature in self._features}

        for sentence in sentences:
            for feature in self._features:
                processed = self.get_features(sentence, feature)
                for item in processed:
                    if lower:
                        item = item.lower()
                        
                    if item not in self._vocab[feature] and counts[feature][item] >= min_count:
                        self._vocab[feature][item] = len(self._vocab[feature])
                    
    def save_vocab(self):
        with open(self.vocabFile, 'w') as f:
            json.dump(self._vocab, f)
            
    def get_features(self, sentence, feature):
        if feature not in derived_features:
            return sentence[feature]

        if feature == 'deps':
            return sentence[feature][-1]
        
        if feature == 'altlex_position':
            return ['<'] * sentence['prev_len'] + ['-'] * sentence['altlex_len'] + ['>'] * sentence['curr_len']

        if feature == 'root_position':
            ret = []
            found_root = False
            for dep in sentence['deps']:
                if dep[-1].lower() == 'root':
                    found_root = True
                    ret.append('-')
                elif not found_root:
                    ret.append('<')
                else:
                    ret.append('>')
            return ret

    def get_indices(self, tokens, feature):
        ret = []
        for token in tokens:
            if token in self.vocab[feature]:
                ret.append(self.vocab[feature][token])
            else:
                ret.append(self.vocab[feature]['UNK'])
        return ret
    
    def get_sentence_indices(self, sentence):
        ret = []
        for feature in self._features:
            tokens = self.get_features(sentence, feature)
            indices = self.get_indices(tokens, feature)
            ret.append(indices)
        return ret

    def get_shortened(self, sentence, MAX_LEN=MAX_LEN):
        tokens = sentence['words']
        start = 0
        end = len(tokens) - 1
        if len(tokens) > MAX_LEN:
            if sentence['prev_len'] < MAX_LEN // 2 - sentence['altlex_len']:
                start = 0
                end = MAX_LEN-1
            elif sentence['curr_len'] < MAX_LEN // 2 - sentence['altlex_len']:
                start = len(tokens) - MAX_LEN
                end = len(tokens) - 1
            else:
                start = sentence['prev_len'] - MAX_LEN // 2
                end = start + MAX_LEN - 1
        return start, end
    
    def preprocess(self, sentences, MAX_LEN=MAX_LEN):
        sentences = list(sentences)
        X = np.zeros(shape=(len(sentences), MAX_LEN, len(self._features)), dtype=int)
        mask = np.zeros(shape=(len(sentences), MAX_LEN))
        
        for sentence_index, sentence in enumerate(sentences):
            features = self.get_sentence_indices(sentence)
            start, end = self.get_shortened(sentence, MAX_LEN)
            for feature_index, tokens in enumerate(features):
                for token_index, token in enumerate(tokens):
                    if token_index > end:
                        break
                    if token_index < start:
                        continue
                    X[sentence_index][token_index-start][feature_index] = token
                    mask[sentence_index][token_index-start] = 1

        return X, mask

    def get_labels(self, sentences, binarize=True):
        return np.array([int(int(i['label']) != 0) for i in sentences])

    def get_ranges(self, sentences):
        starts = []
        ends = []
        
        for index, sentence in enumerate(sentences):
            start, end = self.get_shortened(sentence)
            starts.append(sentence['prev_len']-start)
            ends.append(sentence['prev_len'] + sentence['altlex_len'] - 1 - start)
            if starts[-1] >= MAX_LEN:
                print('s', index, len(sentence['words']), starts[-1], start, end, sentence['prev_len'], sentence['altlex_len'], sentence['curr_len'])
            if ends[-1] >= MAX_LEN:
                print('e', index, len(sentence['words']), ends[-1], start, end, sentence['prev_len'], sentence['altlex_len'], sentence['curr_len'])
                
        return np.array(starts), np.array(ends)

    def filter(self, sentences):
        #remove sentences that fit a certain category
        ret = []
        for sentence in sentences:
            if sentence['prev_len'] + sentence['altlex_len'] + sentence['curr_len'] > len(sentence['words']):
                if sentence['prev_len'] + sentence['altlex_len'] > len(sentence['words']):
                    continue
                sentence['curr_len'] = len(sentence['words']) - (sentence['prev_len'] + sentence['altlex_len'])
            ret.append(sentence)
        return ret
    
    def train(self, train, dev=None, test=None):
        train = self.filter(train)
        X_train, mask_train = self.preprocess(train)
        y_train = self.get_labels(train)
        altlex_start_train, altlex_end_train = self.get_ranges(train)
        
        if dev is not None:
            dev = self.filter(dev)
            X_dev, mask_dev = self.preprocess(dev)
            y_dev = self.get_labels(dev)
            altlex_start_dev, altlex_end_dev = self.get_ranges(dev)
            
        if test is not None:
            test = self.filter(test)
            X_test, mask_test = self.preprocess(test)
            y_test = self.get_labels(test)
            altlex_start_test, altlex_end_test = self.get_ranges(test)
            
        train_lstm((X_train, y_train, mask_train, altlex_start_train, altlex_end_train),
                    self.vocab,
                    self.classifierFile,
                    dev=(X_dev, y_dev, mask_dev, altlex_start_dev, altlex_end_dev),
                    test=(X_test, y_test, mask_test, altlex_start_test, altlex_end_test))

    def predict(self, data):
        X, mask = self.preprocess(data)
        altlex_start, altlex_end = self.get_ranges(data)
        return predict(self.classifier, X, mask, altlex_start, altlex_end)
    
    def findAltlexes(self, sentences):
        #first create new metadata with prev_len, altlex_len, curr_len to every sentence
        data = []
        indices = []
        for index,sentence in enumerate(sentences):
            starts = findAltlexes(sentence['words'], self.causalAltlexes)
            for altlex in starts:
                datum = dict(sentence)
                datum['prev_len'] = starts[altlex]
                datum['altlex_len'] = len(altlex)
                datum['curr_len'] = len(sentence['words']) - datum['prev_len'] - datum['altlex_len']
                data.append(datum)
                indices.append(index)

        predictions = self.predict(data).tolist()

        ranges = collections.defaultdict(list)
        for index, sentence, prediction in zip(indices, data, predictions):
            start = sentence['prev_len']
            end = start + sentence['altlex_len']
            ranges[index].append((prediction, start, end))
        return ranges
        
        
