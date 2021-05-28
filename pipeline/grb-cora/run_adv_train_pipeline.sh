python adv_train_pipeline.py \
--n_epoch 8000 \
--dataset grb-cora \
--feat_norm arctan \
--data_dir ../data/grb-cora \
--model_dir ../saved_models/grb-cora-arctan-ind-adv-fgsm-10/ \
--config_dir ./grb-cora/ \
--dropout 0.5 \
--eval_every 1 \
--save_after 0 \
--early_stop \
--n_train 1 \
--train_mode inductive \
--attack_adv fgsm \
--attack_epoch 10 \
--attack_lr 0.1 \
--n_attack 1 \
--n_inject 60 \
--n_edge_max 20 \
--gpu 0