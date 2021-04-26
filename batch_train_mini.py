# -*- coding：utf-8 -*-
import os
seed_ = 2

for fold_index_ in range(0, 5):
    path = 'file_name'
    print(path)
    os.system("CUDA_VISIBLE_DEVICES=0 python train_minibs.py "
              "--n_epoch 5 "
              "--epochs 15 "
              "--lr 0.0001 "
              "--mcbp mbp "
              "--weight-decay-fc 0 "
              "--loss-type regularizationL1_yL2 "
              "--alpha 0 "
              "--seed {seed} "
              "--num_workers 0 "
              "--fold_index {fold_index} "
              "--resume ./result/{path}/checkpoint/ "
              "--name {path}".format(seed=seed_, fold_index=fold_index_, path=path))
    os.system("CUDA_VISIBLE_DEVICES=0 python test_mul.py "
              "--mcbp mbp "
              "--weight-decay-fc 0 "
              "--seed {seed} "
              "--fold_index {fold_index} "
              "--times 27 27 "
              "--resume ./result/{path}/checkpoint/ "
              "--name {path}".format(seed=seed_, fold_index=fold_index_, path=path))