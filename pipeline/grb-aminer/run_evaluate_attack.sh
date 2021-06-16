python evaluate_attack.py \
--dataset grb-aminer \
--dataset_mode full \
--feat_norm arctan \
--data_dir ../data/grb-aminer/ \
--config_dir ./grb-aminer/ \
--model_dir ../saved_models/grb-aminer-arctan-ind/ \
--model_file "checkpoint.pt" \
--attack_dir ../results/grb-aminer-arctan-ind \
--attack_adj_name "0/adj.pkl" \
--attack_feat_name "0/features.npy" \
--weight_type 'polynomial' \
--save_dir ./exp_results/grb-aminer/ \
--gpu 0