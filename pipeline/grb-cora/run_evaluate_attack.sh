python evaluate_attack.py \
--dataset grb-cora \
--dataset_mode medium \
--feat_norm arctan \
--data_dir ../data/grb-cora/ \
--config_dir ./grb-cora/ \
--model_dir ../saved_models/grb-cora-arctan-ind/ \
--model_file "0/checkpoint.pt" \
--attack_dir ../results/grb-cora-arctan-ind \
--attack_adj_name "0/adj.pkl" \
--attack_feat_name "0/features.npy" \
--weight_type 'polynomial' \
--save_dir ./exp_results/grb-cora/ \
--gpu 0
