python attack_pipeline.py \
--n_epoch 5000 \
--lr 0.01 \
--dataset grb-citeseer \
--dataset_mode easy medium hard full \
--data_dir ../data/grb-citeseer/ \
--config_dir ./grb-citeseer/ \
--feat_norm arctan \
--model gcn \
--model_dir ../saved_models/grb-citeseer-arctan-ind-sur/ \
--model_file 0/checkpoint.pt \
--save_dir ../results/grb-citeseer-arctan-ind/ \
--n_attack 1 \
--n_inject 30 \
--n_edge_max 20 \
--early_stop \
--gpu 0