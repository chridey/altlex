# altlex

Git repository for software associated with the 2016 ACL paper "Identifying Causal Relations Using Parallel Wikipedia Articles."

##Dependencies
flask

gensim

nltk

numpy

requests

scipy

sklearn

MOSES

##How To

For the entire pipeline, start at step 0a using the Simple and English Wikipedia dumps.

Given the data provided with the ACL submission (altlex_train_paraphrases.tsv), parse the paraphrase pairs and start at step 3.

0a) Parse data 

0b) Create word and sentence embeddings using ```gensim``` and save the model file as <embeddings_file>.

1) Find paraphrase pairs from English and Simple Wikipedia 
  1a) Start the embeddings server (it may take a while to load the embeddings and data)
      python altlex/embeddings/representationServer.py
  1b) 
      python altlex/misc/makeWikipediaPairs.py <embeddings_file> <parallel_wikipedia_file> <parsed_wikipedia_directory> <matches_file> <num_processes (optional)> <start_point (optional)>
  1c) Restrict the output to be above the thresholds and make sure all pairs are 1-to-1.
      python altlex/misc/getGreedyMatches.py <matches_file> .69 .65 .75 

2) Format pairs for input to MOSES (version 2.1 was used for all experiments)
python ~/altlex/misc/formatMoses.py parsed_pairs moses/english moses/simple
lmplz -o 3 < english_mod > lm.english_plus
perl train-model.perl --external-bin-dir moses/RELEASE-2.1/binaries/linux-64bit/training-tools/ --corpus corpus/corpus --f simple_plus --e english_plus --root-\
dir . --lm 0:3:/local/nlp/chidey/moses/lm.english_plus -mgiza

3) Determine possible new altlexes by using the word alignments to determine phrases that align with known connectives
#for trinary
python ~/altlex/misc/alignAltlexes.py parsed_pairs moses/model/aligned.grow-diag-final wash_plus 1

4) Make KLD weights
python ~/altlex/misc/calcKLDivergence.py parsed_pairs moses/model/aligned.grow-diag-final aligned_labeled/wash_plus_initLabels_mod.json.gz wash_plus.kldt

5) Make feature set
python ~/altlex/misc/extractFeatures.py parsed_pairs moses/model/aligned.grow-diag-final aligned_labeled/wash_plus_initLabels_mod.json.gz features.json.gz featureExtractor.config.json

6) Train model

6a) Train model with bootstrapping

(see the ablation directory for example commands run)

##Data Format

###Parallel Wikipedia Format

This is a gzipped, JSON-formatted file.  The "titles" array is the shared title name of the English and Simple Wikipedia articles.  The "articles" array consists of two arrays and each of those arrays must be the same length as the "titles" array and the indices into these arrays must point to the aligned articles and titles.  Each article within the articles array is an array of tokenized sentence strings (but not word tokenized).

The format of the dictionary is as follows:
```
{'files': [english_name, simple_name],

 'articles': [
 
              [[article_1_sentence_1_string, article_1_sentence_2_string, ...],
              
               [article_2_sentence_1_string, article_2_sentence_2_string, ...],
               
               ...
               
              ],
              
              [[article_1_sentence_1_string, article_1_sentence_2_string, ...],
              
               [article_2_sentence_1_string, article_2_sentence_2_string, ...],
               
               ...
               
              ]
              
             ],
  
  titles': [title_1_string, title_2_string, ...]

}
```

### Parsed Pairs Format

This is a gzipped, JSON-formatted list of parsed sentences.  Paraphrase pairs are consecutive even and odd indices.

The data is formatted as follows:
[
  {
   'dep': [[[governor_index, dependent_index, relation_string], ...], ...], 
   'lemmas': [[lemma_1_string, lemma_2_string, ...], ...],
   'pos': [[pos_1_string, pos_2_string, ...], ...],
   'parse': [parenthesized_parse_1_string, ...], 
   'words': [[word_1_string, word_2_string, ...], ...] , 
   'ner': [[ner_1_string, ner_2_string, ...], ...]
  },
  ...
]

