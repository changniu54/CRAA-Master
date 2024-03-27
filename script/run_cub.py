#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chang Niu
"""


## Generalized CI-ZSL
## --class-embedding sent for sent_splits.mat --class_embedding att for att_splits.mat
import os
os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4  python craa.py --task_mode gzsl \
--manualSeed 3483 --preprocessing --cuda --image_embedding res101 --class_embedding att \
--ngh 2048 --ndh 2048 --lr_b 0.0001 --lr_i 0.0001 --lambda1 10 --critic_iter 5 --dataset CUB \
--nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 \
--seen_class_num 150 --base_class_num 30 --task_num 5 \
--center_margin 200 --incenter_weight 0.8 \
--protoSize 1024 --hSize 2048 --gamma 30 --epsilon 3 \
--nepoch_base 50 --nepoch_incremental 50 --refine_fea \
--syn_replay_num 40 --syn_previous_seen_num 40 --syn_unseen_num 10''')


# ## CI-ZSL for Unseen
# ## --class-embedding sent for sent_splits.mat --class_embedding att for att_splits.mat
# import os
# os.system('''CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4  python craa.py --task_mode zsl \
# --manualSeed 3483 --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --ngh 2048 --ndh 2048 --lr_b 0.0001 --lr_i 0.0001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 \
# --seen_class_num 150 --base_class_num 30 --task_num 5 \
# --center_margin 200 --incenter_weight 0.8 \
# --protoSize 1024 --hSize 2048 --gamma 30 --epsilon 3 \
# --nepoch_base 50 --nepoch_incremental 50 --refine_fea \
# --syn_replay_num 40 --syn_previous_seen_num 40 --syn_unseen_num 30''')



# ## Incremental Learning for Seen
# ## --class-embedding sent for sent_splits.mat --class_embedding att for att_splits.mat
# import os
# os.system('''CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4  python craa.py --task_mode il \
# --manualSeed 3483 --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --ngh 2048 --ndh 2048 --lr_b 0.0001 --lr_i 0.0001 --lambda1 10 --critic_iter 5 --dataset CUB \
# --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 \
# --seen_class_num 150 --base_class_num 30 --task_num 5 \
# --center_margin 200 --incenter_weight 0.8 \
# --protoSize 1024 --hSize 2048 --gamma 30 --epsilon 5 \
# --nepoch_base 50 --nepoch_incremental 50 --refine_fea \
# --syn_replay_num 40 --syn_previous_seen_num 40 ''')