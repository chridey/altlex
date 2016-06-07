echo 'full_with_connective'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod.json.gz --start 1 --tunefile features_full_plus_orig_mod_test.json.gz --testfile features_full_plus_orig_mod_test.json.gz  --batch_size 3000 --num_epochs 200 --verbose --ablate lemmas,stems,adv.all,adj.all --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'interaction_with_connective'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod.json.gz --start 1 --tunefile features_full_plus_orig_mod_test.json.gz --testfile features_full_plus_orig_mod_test.json.gz  --batch_size 3000 --num_epochs 200 --interaction --verbose --ablate lemmas,stems,adv.all,adj.all --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'connective_only'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod.json.gz --start 1 --tunefile features_full_plus_orig_mod_test.json.gz --testfile features_full_plus_orig_mod_test.json.gz  --batch_size 3000 --num_epochs 200 --verbose --filter connective --ablate lemmas,stems --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'connective_with_pos'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod.json.gz --start 1 --tunefile features_full_plus_orig_mod_test.json.gz --testfile features_full_plus_orig_mod_test.json.gz  --batch_size 3000 --num_epochs 200 --verbose --filter connective_lemmas_pos --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'connective_unistems'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod.json.gz --start 1 --tunefile features_full_plus_orig_mod_test.json.gz --testfile features_full_plus_orig_mod_test.json.gz  --batch_size 3000 --num_epochs 200 --verbose --filter connective_unistems --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'connective_bistems'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod.json.gz --start 1 --tunefile features_full_plus_orig_mod_test.json.gz --testfile features_full_plus_orig_mod_test.json.gz  --batch_size 3000 --num_epochs 200 --verbose --filter connective_bistems --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'

echo 'connective_all_stems'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod.json.gz --start 1 --tunefile features_full_plus_orig_mod_test.json.gz --testfile features_full_plus_orig_mod_test.json.gz  --batch_size 3000 --num_epochs 200 --verbose --filter stems --search_parameters '{"penalty": ["l2", "elasticnet"], "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]}'