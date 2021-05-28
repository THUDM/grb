python train_pipeline.py \
--n_epoch 8000 \
--dataset grb-cora \
--feat_norm arctan \
--data_dir ../data/grb-cora \
--model_dir ../saved_models/grb-cora-arctan-ind/ \
--config_dir ./grb-cora/ \
--dropout 0.5 \
--eval_every 1 \
--save_after 0 \
--early_stop \
--n_train 1 \
--train_mode inductive \
--gpu 0

# Train surrogate model
python train_pipeline.py \
--n_epoch 8000 \
--dataset grb-cora \
--feat_norm arctan \
--data_dir ../data/grb-cora \
--model_dir ../saved_models/grb-cora-arctan-ind-sur/ \
--config_dir ./grb-cora/ \
--dropout 0.5 \
--eval_every 1 \
--save_after 0 \
--early_stop \
--n_train 1 \
--train_mode inductive \
--gpu 0