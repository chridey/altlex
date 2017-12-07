import sys

import math
import numpy as np
import gensim

from sklearn import metrics
from lasagne.layers import set_all_param_values
import lasagne

from altlex.neural.altlexLSTM import AltlexLSTM

features = ('words', 'pos', 'deps', 'frames', 'altlex_position', 'root_position')
dimensions = {'words': 100,
              'pos': 20,
              'deps': 20,
              'frames': 50,
              'altlex_position': 3,
              'root_position': 3}

WEIGHT_FACTOR = 1
EMBED_DIM = 100
ERROR_ANALYSIS = False

def get_rand_indices(N, batch_size, num_batchs):
    # create indexs for training batchs
    idxs = np.random.choice(N, (1, N), replace=False)
    idx_groups = [idxs[0, i*batch_size:(i+1)*batch_size] for i in range(num_batchs)]
    # append remaining indexes that are not enough to form a batch
    idx_groups.append(idxs[0, num_batchs*batch_size:])
    return idx_groups

# split training data points into mini batches
def split_data(X, Y, mask, altlex_start, altlex_end, idx_groups):
    lem_list, pos_list, dep_list, frm_list, alt_list, dir_list = [], [], [], [], [], []
    m_list, y_list, w_list, as_list, ae_list, r_list = [], [], [], [], [], []
    for idx in idx_groups:
        lem_list.append(X[idx, :, 0])
        pos_list.append(X[idx, :, 1])
        dep_list.append(X[idx, :, 2])
        frm_list.append(X[idx, :, 3])
        alt_list.append(X[idx, :, 4])
        dir_list.append(X[idx, :, 5])
        m_list.append(mask[idx, :])
        gold = Y[idx]
        r, w = compute_weight(gold)
        y_list.append(gold)
        w_list.append(w)
        as_list.append(altlex_start[idx])
        ae_list.append(altlex_end[idx])
        r_list.append(r)
    return (lem_list, pos_list, dep_list, frm_list, alt_list, dir_list, 
            y_list, m_list, w_list, as_list, ae_list, r_list)

def count_match(l1, l2):
    if len(l1) != len(l2):
        print "output length mismatch..."
    count = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            count += 1
    return count

def get_error_indices(y1, y2):
    if len(y1) != len(y2):
        print "output length mismatch..."
    ret = []
    for i in range(len(y1)):
        if y1[i] != y2[i]:
            ret.append(i+1)
    return ret

def find_most_common_errors(error_lists, cutoff):
    d = {}
    for error_list in error_lists:
        for idx in error_list:
            if idx in d:
                d[idx] += 1
            else:
                d[idx] = 1
    indices = d.keys()
    indices.sort(key=lambda x: d[x], reverse=True)
    return [(idx, d[idx]) for idx in indices if d[idx] >= cutoff]

def compute_weight(gold):
    r = np.count_nonzero(gold) / float(len(gold))
    multiplier = int(math.floor(1/r))*WEIGHT_FACTOR
    w = [1 if label == 0 else multiplier for label in gold]
    return r, np.asarray(w)

def validate(lstm, i, X, y, mask, start, end):
    s = 'Epoch: {:1d}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}, Accuracy: {:.3f}'
    scores = lstm.predict(X[:, :, 0], 
                          X[:, :, 1], 
                          X[:, :, 2], 
                          X[:, :, 3], 
                          X[:, :, 4], 
                          X[:, :, 5], 
                          mask, 
                          start,
                          end).tolist()
    y_pred = np.asarray([1 if score > 0.5 else 0 for score in scores])
    # calculate metric scores
    p = metrics.precision_score(y, y_pred)
    r = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    acc = float(count_match(y, y_pred))/len(y)
    print(s.format(i, p, r, f1, acc))
    error_list = list(get_error_indices(y, y_pred))
    if ERROR_ANALYSIS:
        print('Error terms: '+str(error_list))
    return p, r, f1, acc, error_list

def evaluate(lstm, X, y, mask, start, end):
    s = 'Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f},, Accuracy: {:.3f}'
    scores = lstm.predict(X[:, :, 0], 
                          X[:, :, 1], 
                          X[:, :, 2], 
                          X[:, :, 3], 
                          X[:, :, 4], 
                          X[:, :, 5], 
                          mask, 
                          start,
                          end).tolist()
    y_pred = np.asarray([1 if score > 0.5 else 0 for score in scores])
    p = metrics.precision_score(y, y_pred)
    r = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    acc = float(count_match(y_pred, y))/len(y)
    print(s.format(p, r, f1, acc))

def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = param_values = [params[i] for i in param_keys]
    set_all_param_values(model.network, param_values)

def match_embeddings(model, vocabs):
    embeddings = [None for i in range(len(vocabs))]
    for lemma in vocabs.keys():   
        idx = vocabs[lemma]   
        if lemma in model:
            embeddings[idx] = model[lemma]
        else:
            embeddings[idx] = np.random.uniform(-.2, .2, EMBED_DIM)
    return np.array(embeddings)

def train(train, vocab, param_file, dev=None, test=None, batch_size=1000, num_epochs=30):
    embed_caps = [dimensions[i] for i in features if i in vocab]
    d = 100
    cost_thres = 1e-3

    X_train, y_train, mask_train, altlex_start_train, altlex_end_train = train
    N = X_train.shape[0]
    M = X_train.shape[1]
    D = [len(vocab[i]) for i in features if i in vocab]
    
    do_dev = False
    if dev is not None:
        X_dev, y_dev, mask_dev, altlex_start_dev, altlex_end_dev = dev
        r_dev, _ = compute_weight(y_dev)        
        do_dev = True
        
    do_test = False
    if test is not None:
        X_test, y_test, mask_test, altlex_start_test, altlex_end_test = test
        r_test, _ = compute_weight(y_test)
        do_test = True

    # initialize pretrained word embeddings
    model = gensim.models.KeyedVectors.load_word2vec_format('/proj/nlp/users/ethan/glove_word2vec_100d.txt', binary=False)
    We = match_embeddings(model, vocab['words'])

    num_batchs = N / batch_size

    print('N: ' + str(N) + ' M: ' + str(M) + ' Dev: ' + str(r_dev) + ' Test: ' + str(r_test))
    print('Batch size: ' + str(batch_size) + ' Num batch: ' + str(num_batchs+1))
    print('Embedding dim: ' + str(D) + ' Caps: ' + str(embed_caps))

    print('----------starting to train network-----------')

    train_str = 'Epoch: {:1d}, Batch: {:1d}, Cost: {:.5f}, Ratio: {:.3f}'
    lstm = AltlexLSTM(M, D, embed_caps, d, We)
    best_score = 0
    best_f1 = 0
    best_epoch = 0
    stop = False

    error_lists = []

    for i in range(num_epochs):

        # get random indices to split data
        idx_groups = get_rand_indices(N, batch_size, num_batchs)
        # split data into batches
        l = split_data(X_train, y_train, mask_train, altlex_end_train, altlex_end_train, idx_groups)

        # train each batch
        min_cost = 1
        for j in range(num_batchs+1):
            cost = lstm.train(l[0][j], 
                              l[1][j], 
                              l[2][j], 
                              l[3][j], 
                              l[4][j], 
                              l[5][j], 
                              l[6][j], 
                              l[7][j], 
                              l[8][j],
                              l[9][j],
                              l[10][j])
            print(train_str.format(i, j, float(cost), l[-1][j]))
            min_cost = min(min_cost, cost)
            # stop if cost is too low or nan
            min_cost = min(min_cost, cost)
            if math.isnan(cost):
                stop = True
                break
        
        if min_cost < cost_thres:
            stop = True

        if do_dev:
            p, r, f1, acc, errors = validate(lstm, i, X_dev, y_dev, mask_dev, altlex_start_dev, altlex_end_dev)
        else:
            f1 = 0
            
        error_lists.append(errors)
        score = f1

        # save best parameters
        if score >= best_score:
            best_score = score
            best_f1 = f1
            best_epoch = i
            lstm.save_params(param_file)

        # stop if cost is too low or nan
        if stop:
            break
    
    if ERROR_ANALYSIS:        
        print('----------errors from dev set-----------')
        print('Most Common Errors: '+str(find_most_common_errors(error_lists, 2)))
    
    print('----------final evaluation on testset-----------')

    # run test set
    load(lstm, param_file+'.npz')
    print('Loaded parameters from epoch ' + str(best_epoch) + ' (Dev F1 = ' + str(best_f1) + ')')
    if do_test:
        evaluate(lstm, X_test, y_test, mask_test, altlex_start_test, altlex_end_test)

if __name__ == '__main__':
    main()

