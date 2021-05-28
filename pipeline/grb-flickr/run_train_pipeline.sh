python train_pipeline.py \
--n_epoch 8000 \
--dataset grb-flickr \
--feat_norm linearize \
--data_dir ../data/grb-flickr \
--model_dir ../saved_models/grb-flickr-linear-ind/ \
--config_dir ./grb-flickr/ \
--dropout 0.5 \
--eval_every 10 \
--save_after 0 \
--early_stop \
--n_train 1 \
--train_mode inductive \
--gpu 0