echo 'most common class'

echo 'connective only'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1  --testfile features_full_plus_orig_mod_binary_test.json.gz --batch_size 3000 --combined --num_epochs 100  --filter connective --ablate lemmas,stems --parameters '{"penalty": "elasticnet", "alpha": 0.01}' --save ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_connective_only_final

echo 'full_with_connective'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1  --testfile features_full_plus_orig_mod_binary_test.json.gz --batch_size 3000 --combined --num_epochs 100   --ablate lemmas,stems --parameters '{"penalty": "elasticnet", "alpha": 0.01}' --save ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_full_with_connective_final

echo 'full'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1  --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 100  --ablate connective --parameters '{"penalty": "l2", "alpha": 0.00001}' --save ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_final

echo 'kld'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1  --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 100  --filter kld --parameters '{"penalty": "l2", "alpha": 0.001}' --save ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_kld_only_final

echo 'semantic'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1  --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 100  --ablate kld,connective --parameters '{"penalty": "elasticnet", "alpha": 0.0001}' --save ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_semantic_only_final

echo 'semantic_interaction'
python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1  --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 100  --ablate kld,connective --interaction --parameters '{"penalty": "elasticnet", "alpha": 0.0001}' --save ../models/final/full_plus_sgd_st1_inter1_unbalanced_combined_semantic_only_final

echo 'full_interaction'
#python /home/chidey/altlex/misc/trainFeatureWeights.py features_full_plus_orig_mod_binary.json.gz --start 1  --testfile features_full_plus_orig_mod_binary_test.json.gz  --batch_size 3000 --combined --num_epochs 200 --interaction --verbose --ablate connective --parameters '{"penalty": "elasticnet", "alpha": 0.0001}' --save ../models/final/full_plus_sgd_st1_inter1_unbalanced_combined_final