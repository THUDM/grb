# Attack pipeline
# full
python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-aminer \
--dataset_mode full \
--data_dir ../data/grb-aminer/ \
--feat_norm arctan \
--config_dir ./grb-aminer/ \
--model gcn \
--model_dir ../saved_models/grb-aminer-arctan-ind-sur/ \
--model_file "0/checkpoint.pt" \
--save_dir ../results/grb-aminer-arctan-ind/ \
--n_attack 1 \
--n_inject 1500 \
--n_edge_max 100 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1

# easy
python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-aminer \
--dataset_mode easy \
--data_dir ../data/grb-aminer/ \
--feat_norm arctan \
--config_dir ./grb-aminer/ \
--model gcn \
--model_dir ../saved_models/grb-aminer-arctan-ind-sur/ \
--model_file "0/checkpoint.pt" \
--save_dir ../results/grb-aminer-arctan-ind/ \
--n_attack 1 \
--n_inject 500 \
--n_edge_max 100 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1

# medium
python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-aminer \
--dataset_mode medium \
--data_dir ../data/grb-aminer/ \
--feat_norm arctan \
--config_dir ./grb-aminer/ \
--model gcn \
--model_dir ../saved_models/grb-aminer-arctan-ind-sur/ \
--model_file "0/checkpoint.pt" \
--save_dir ../results/grb-aminer-arctan-ind/ \
--n_attack 1 \
--n_inject 500 \
--n_edge_max 100 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1

# hard
python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-aminer \
--dataset_mode hard \
--data_dir ../data/grb-aminer/ \
--feat_norm arctan \
--config_dir ./grb-aminer/ \
--model gcn \
--model_dir ../saved_models/grb-aminer-arctan-ind-sur/ \
--model_file "0/checkpoint.pt" \
--save_dir ../results/grb-aminer-arctan-ind/ \
--n_attack 1 \
--n_inject 500 \
--n_edge_max 100 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1
