# altlex

Git repository for software associated with the 2016 ACL paper "Identifying Causal Relations Using Parallel Wikipedia Articles."

##Dependencies
```
flask
gensim
nltk
numpy
requests
scipy
sklearn
```

MOSES

KenLM

##How To

For the entire pipeline, start at step 1.i using the Simple and English Wikipedia dumps in the format "Parallel Wikipedia Format".

Given the data provided with the ACL submission (```altlex_train_paraphrases.tsv```), parse the paraphrase pairs in the format "Parsed Pairs Format", save the files into ```$parsed_pairs_directory```, and start at step 3.

0. Preprocess data
    1. Parse data and save in the format "Parsed Wikipedia Format" in the directory ```$parsed_wikipedia_directory``` (there can be multiple files in this directory).
    2. Create word and sentence embeddings using ```gensim``` and save the model file as ```$embeddings_file```.

1. Find paraphrase pairs from English and Simple Wikipedia 
    1. Start the embeddings server (it may take a while to load the embeddings and data):
      
      ```python altlex/embeddings/representationServer.py```

    2. Determine possible paraphrase pairs and create the file ```$matches_file```:
      
      ```python altlex/misc/makeWikipediaPairs.py $embeddings_file $parallel_wikipedia_file $parsed_wikipedia_directory $matches_file $num_processes (optional) $start_point (optional)```
    
    3. Restrict the output to be above the thresholds and make sure all pairs are 1-to-1.
      
      ```python altlex/misc/getGreedyMatches.py $matches_file $min_doc2vec_score $min_wiknet_score $wiknet_penalty > $reduced_matches_file ```
      
    4. Create the directory ```$parsed_pairs_directory``` with files of the format "Parsed Pairs Format" using the output of 1c.
  
2. Format pairs for input to MOSES (version 2.1 was used for all experiments):
   
  ```
  python altlex/misc/formatMoses.py $parsed_pairs_directory corpus/$english_sentences corpus/$simple_sentences```
  lmplz -o 3 < $english_sentences > $english_language_model
  perl train-model.perl --external-bin-dir moses/RELEASE-2.1/binaries/linux-64bit/training-tools/ --corpus corpus --f $english_sentences --e $simple_sentences --root-dir . --lm 0:3:$english_language_model -mgiza
  ```

3. Determine possible new altlexes by using the word alignments to determine phrases that align with known connectives 
   (```$binary_flag``` should be 1 if predicting causal vs non-causal and 0 if predicting causal vs reason vs result):
   
  ```python altlex/misc/alignAltlexes.py $parsed_pairs_directory model/aligned.grow-diag-final $session_name $binary_flag```

4. Make KLD weights
   
  ```python altlex/misc/calcKLDivergence.py <parsed_pairs_directory> model/aligned.grow-diag-final ${session_name}_initLabels.json.gz ${kld_name}.kldt $binary_flag```

5. Make feature set (see "Feature Extractor Config Format" for $json_config)
  
  ```python altlex/misc/extractFeatures.py parsed_pairs model/aligned.grow-diag-final ${session_name}_initLabels.json.gz $features_file  $json_config```

6. Train model (see the ablation directory for example commands run)
   
  ```python altlex/misc/trainFeatureWeights.py $features_file```

7. Train model with bootstrapping

  ```python altlex/misc/bootstrapping.py $features_file $parsed_pairs_directory ${session_name}_initLabels.json.gz```

##Data Format

###Parallel Wikipedia Format

This is a gzipped, JSON-formatted file.  The "titles" array is the shared title name of the English and Simple Wikipedia articles.  The "articles" array consists of two arrays and each of those arrays must be the same length as the "titles" array and the indices into these arrays must point to the aligned articles and titles.  Each article within the articles array is an array of tokenized sentence strings (but not word tokenized).

The format of the dictionary is as follows:
```
{"files": [english_name, simple_name],
 "articles": [
              [[article_1_sentence_1_string, article_1_sentence_2_string, ...],
               [article_2_sentence_1_string, article_2_sentence_2_string, ...],
               ...
              ],
              [[article_1_sentence_1_string, article_1_sentence_2_string, ...],
               [article_2_sentence_1_string, article_2_sentence_2_string, ...],
               ...
              ]
             ],
  "titles": [title_1_string, title_2_string, ...]
}
```

### Parsed Wikipedia Format
This is a gzipped, JSON-formatted list of parsed Wikipedia article pairs.  The list stored at 'sentences' is of length 2 and stores each version of the English and Wikipedia article with the same title.

The data is formatted as follows:
```
[
 {
  "index": article_index,
  "title": article_title_string,
  "sentences": [[parsed_sentence_1, parsed_sentence_2, ...],
                [parsed_sentence_1, parsed_sentence_2, ...]
               ]
 },
 ...
]
```

### Parsed Pairs Format

This is a gzipped, JSON-formatted list of parsed sentences.  Paraphrase pairs are consecutive even and odd indices.
For the parsed sentence, see "Parsed Sentence Format."

The data is formatted as follows:
```
[
  ...,
  parsed_sentence_2,
  parsed_sentence_3,
  ...
]
```

### Parsed Sentence Format

Each parsed sentence is of the following format:
```
{
   "dep": [[[governor_index, dependent_index, relation_string], ...], ...], 
   "lemmas": [[lemma_1_string, lemma_2_string, ...], ...],
   "pos": [[pos_1_string, pos_2_string, ...], ...],
   "parse": [parenthesized_parse_1_string, ...], 
   "words": [[word_1_string, word_2_string, ...], ...] , 
   "ner": [[ner_1_string, ner_2_string, ...], ...]
}
```  

### Feature Extractor Config Format
```
{"framenetSettings": 
   {"binary": true/false}, 
 "featureSettings": 
   {
   "arguments_cat_curr": true/false, 
   "arguments_verbnet_prev": true/false, 
   "head_word_cat_curr": true/false, 
   "head_word_verbnet_prev": true/false, 
   "head_word_verbnet_altlex": true/false, 
   "head_word_cat_prev": true/false, 
   "head_word_cat_altlex": true/false, 
   "kld_score": true/false, 
   "head_word_verbnet_curr": true/false, 
   "arguments_verbnet_curr": true/false, 
   "framenet": true/false, 
   "arguments_cat_prev": true/false, 
   "connective": true/false
   }, 
 "kldSettings": 
   {"kldDir": $kld_name}
}
```
