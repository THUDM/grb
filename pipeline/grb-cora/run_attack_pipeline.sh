python attack_pipeline.py \
--n_epoch 5000 \
--lr 0.01 \
--dataset grb-cora \
--dataset_mode easy medium hard \
--data_dir ../data/grb-cora/ \
--config_dir ./grb-cora/ \
--feat_norm arctan \
--model gcn \
--model_dir ../saved_models/grb-cora-arctan-ind-sur/ \
--model_file 0/checkpoint.pt \
--save_dir ../results/grb-cora-arctan-ind/ \
--n_attack 1 \
--n_inject 20 \
--n_edge_max 20 \
--early_stop \
--gpu 0

python attack_pipeline.py \
--n_epoch 5000 \
--lr 0.01 \
--dataset grb-cora \
--dataset_mode full \
--data_dir ../data/grb-cora/ \
--config_dir ./grb-cora/ \
--feat_norm arctan \
--model gcn \
--model_dir ../saved_models/grb-cora-arctan-ind-sur/ \
--model_file 0/checkpoint.pt \
--save_dir ../results/grb-cora-arctan-ind/ \
--n_attack 1 \
--n_inject 60 \
--n_edge_max 20 \
--gpu 0