echo 'most common class'
python /home/chidey/altlex/evaluation/evaluation.py --common --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  ../aligned_labeled/features_full_plus_orig_mod_binary.json.gz --format latex

echo 'most common class after bootstrapping'
python /home/chidey/altlex/evaluation/evaluation.py --common --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  ../models/final/full_plus_sgd_st1_inter1_unbalanced_combined_bootstrap.1.train --format latex

echo 'connective only'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  ../models/final/full_plus_sgd_st1_inter0_balanced_combined_connective_only_final --format latex

echo 'full_with_connective'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz ../models/final/full_plus_sgd_st1_inter0_balanced_combined_full_with_connective_final --format latex

echo 'full_with_connective'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz ../models/final/full_plus_sgd_st1_inter0_balanced_combined_connective_pos_only_final --format latex

echo 'full'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_final --format latex

echo 'kld'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_kld_only_final --format latex

echo 'semantic'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz   ../models/final/full_plus_sgd_st1_inter0_unbalanced_combined_semantic_only_final --format latex

echo 'semantic_interaction'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz ../models/final/full_plus_sgd_st1_inter1_unbalanced_combined_semantic_only_final --format latex

echo 'full_interaction'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz ../models/final/full_plus_sgd_st1_inter1_unbalanced_combined_final --format latex

echo 'after 1 round of bootstrapping'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz ../models/final/full_plus_sgd_st1_inter1_unbalanced_combined_bootstrap.0 --format latex

echo 'after 2 rounds of bootstrapping'
python /home/chidey/altlex/evaluation/evaluation.py --combined ../models/final/features_full_plus_orig_mod_binary_gold.json.gz  ../models/final/full_plus_sgd_st1_inter1_unbalanced_combined_bootstrap.1 --format latex