echo 'full'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  --batch_size 3000 --num_epochs 200 --verbose --ablate connective --combined  --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'framenet'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  --batch_size 3000 --num_epochs 200 --verbose --ablate connective --combined  --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}' --filter framenet --ablate connective

echo 'wordnet'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  --batch_size 3000 --num_epochs 200 --verbose --ablate connective --combined  --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}' --filter cat --ablate connective

echo 'verbnet'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  --batch_size 3000 --num_epochs 200 --verbose --ablate connective --combined  --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}' --filter verbnet --ablate connective

echo 'wordnet_interaction'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  --batch_size 3000 --num_epochs 200 --verbose --ablate connective --combined  --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}' --filter cat --ablate connective --interaction

echo 'verbnet_interaction'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  --batch_size 3000 --num_epochs 200 --verbose --ablate connective --combined  --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}' --filter verbnet --ablate connective --interaction




