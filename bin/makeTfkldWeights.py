import sys
from chnlp.utils.readers import msrParaphraseReader
from chnlp.ml.tfkld import TfkldTransformer

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    mpr = msrParaphraseReader.MsrParaphraseReader(infile)
    tfkld = TfkldTransformer(verbose=True)
    tfkld.fit(*mpr.asTrainingSet())
    tfkld.save(outfile)
    #joblib.dump(tfkld, outfile)
