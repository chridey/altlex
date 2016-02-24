import json
import base64
import logging

import numpy as np

from flask import Flask, request

from altlex.embeddings import sentenceRepresentation

logger = logging.getLogger(__file__)
app = Flask(__name__)

models = {}

def getEmbeddings(data, model):
    if (data, model) not in models:
        models[(data, model)] = sentenceRepresentation.factory(data, model)
    return models[(data, model)]

@app.route("/titles/")
def titles():
    data = request.args.get('data')
    model = request.args.get('model')    
    return json.dumps(getEmbeddings(data, model).titles)

@app.route("/sentences/")
def sentences():
    data = request.args.get('data')
    model = request.args.get('model')
    articleIndex = int(request.args.get('articleIndex'))
    fileIndex = int(request.args.get('fileIndex'))
    return json.dumps(getEmbeddings(data, model).lookupSentences(articleIndex, fileIndex))

@app.route("/embeddings/")
def embeddings():
    data = request.args.get('data')
    model = request.args.get('model')
    articleIndex = int(request.args.get('articleIndex'))
    fileIndex = int(request.args.get('fileIndex'))
    return json.dumps(np.asarray(getEmbeddings(data, model).lookupEmbeddings(articleIndex, fileIndex)).tolist())

@app.route("/pairEmbeddings/")
def pairEmbeddings():
    data = request.args.get('data')
    model = request.args.get('model')
    articleIndex = int(request.args.get('articleIndex'))
    fileIndex = int(request.args.get('fileIndex'))
    return json.dumps(np.asarray(getEmbeddings(data, model).lookupPairEmbeddings(articleIndex, fileIndex)).tolist())

@app.route("/infer/")
def infer():
    data = request.args.get('data')
    model = request.args.get('model')
    words = request.args.get('words')
    words = base64.b64decode(words).split()
    return json.dumps(np.asarray(getEmbeddings(data, model).infer(words)).tolist())

@app.route("/batchSimilarity/")
def batchSimilarity():
    data = request.args.get('data')
    model = request.args.get('model')
    articleIndex = int(request.args.get('articleIndex'))
    n_jobs = int(request.args.get('njobs'))
    pairs = bool(request.args.get('pairs'))    
    return json.dumps(np.asarray(getEmbeddings(data, model).batchSimilarity(articleIndex,
                                                                            n_jobs=n_jobs,
                                                                            pairs=pairs)).tolist())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='server for parallel wikipedia data and embeddings')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='localhost')
    args = parser.parse_args()
                        
    logging.basicConfig(level=logging.DEBUG)
    app.run(host=args.host, threaded=True, port=args.port)
