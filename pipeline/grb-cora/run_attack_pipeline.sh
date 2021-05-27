python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-cora \
--dataset_mode hard \
--data_dir ../data/grb-cora/ \
--config_dir ./grb-cora/ \
--feat_norm arctan \
--model gcn \
--model_dir ../saved_models/grb-cora-arctan-ind-sur/ \
--model_file "checkpoint.pt" \
--save_dir ../results/grb-cora-arctan-ind/ \
--n_attack 1 \
--n_inject 20 \
--n_edge_max 20 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1