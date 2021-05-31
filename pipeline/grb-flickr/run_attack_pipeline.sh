python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-flickr \
--dataset_mode full \
--data_dir ../data/grb-flickr/ \
--config_dir ./grb-flickr/ \
--feat_norm arctan \
--model_dir ../saved_models/grb-flickr-arctan-ind-sur/ \
--model_file 0/checkpoint.pt \
--model gcn \
--save_dir ../results/grb-flickr-arctan-ind/ \
--n_attack 1 \
--n_inject 600 \
--n_edge_max 100 \
--early_stop \
--gpu 0

python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-flickr \
--dataset_mode easy medium hard \
--data_dir ../data/grb-flickr/ \
--config_dir ./grb-flickr/ \
--feat_norm arctan \
--model_dir ../saved_models/grb-flickr-arctan-ind-sur/ \
--model_file 0/checkpoint.pt \
--model gcn \
--save_dir ../results/grb-flickr-arctan-ind/ \
--n_attack 1 \
--n_inject 200 \
--n_edge_max 100 \
--early_stop \
--gpu 2

