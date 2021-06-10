# Attack pipeline
# full
python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-cora \
--dataset_mode full \
--data_dir ../data/grb-cora/ \
--config_dir ./grb-cora/ \
--feat_norm arctan \
--model gcn \
--model_dir ../saved_models/grb-cora-arctan-ind-sur/ \
--model_file "checkpoint.pt" \
--save_dir ../results/grb-cora-arctan-ind/ \
--n_attack 10 \
--n_inject 60 \
--n_edge_max 20 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1

# easy
python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-cora \
--dataset_mode easy \
--data_dir ../data/grb-cora/ \
--config_dir ./grb-cora/ \
--feat_norm arctan \
--model gcn \
--model_dir ../saved_models/grb-cora-arctan-ind-sur/ \
--model_file "checkpoint.pt" \
--save_dir ../results/grb-cora-arctan-ind/ \
--n_attack 10 \
--n_inject 20 \
--n_edge_max 20 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1

# medium

python attack_pipeline.py \
--n_epoch 2000 \
--lr 0.01 \
--dataset grb-aminer \
--dataset_mode medium \
--data_dir /home/stanislas/Research/GRB/data/grb-aminer/ \
--config_dir /home/stanislas/Research/GRB/pipeline/grb-aminer/ \
--feat_norm arctan \
--model gcn \
--model_dir /home/stanislas/Research/GRB/saved_models/grb-aminer-arctan-ind-sur/ \
--model_file 0/checkpoint.pt \
--save_dir /home/stanislas/Research/GRB/results/grb-aminer-arctan-ind/ \
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
--data_dir /home/stanislas/Research/GRB/data/grb-aminer/ \
--config_dir /home/stanislas/Research/GRB/pipeline/grb-aminer/ \
--feat_norm arctan \
--model gcn \
--model_dir /home/stanislas/Research/GRB/saved_models/grb-aminer-arctan-ind-sur/ \
--model_file 0/checkpoint.pt \
--save_dir /home/stanislas/Research/GRB/results/grb-aminer-arctan-ind/ \
--n_attack 1 \
--n_inject 500 \
--n_edge_max 100 \
--feat_lim_min -1 \
--feat_lim_max 1 \
--gpu 1
