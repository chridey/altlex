echo 'full'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --verbose --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'framenet'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --filter framenet --verbose --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'kld'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --filter kld --verbose --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'semantic'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --ablate kld --verbose --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'interaction'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --interaction --verbose --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'semantic_interaction'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --ablate kld --interaction --verbose --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'semantic_interaction_minus_framenet'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1 --tunefile features_full_plus_orig_mod_binary_test.json.gz --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --ablate kld,framenet --interaction --verbose --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'



