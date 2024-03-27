#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chang Niu
"""


# ## Generalized CI-ZSL
# import os
# os.system('''CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4  python craa.py --task_mode gzsl \
# --manualSeed 806 --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --ngh 2048 --ndh 2048 --lr_b 0.0002 --lr_i 0.0002 --lambda1 10 --critic_iter 5 --dataset FLO \
# --nclass_all 102 --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 \
# --seen_class_num 82 --base_class_num 18 --task_num 5 \
# --center_margin 200 --incenter_weight 0.8 \
# --protoSize 1024 --hSize 2048 --gamma 120 --epsilon 3 \
# --nepoch_base 90 --nepoch_incremental 50 --refine_fea \
# --syn_replay_num 90 --syn_previous_seen_num 90 --syn_unseen_num 10''')


# ## CI-ZSL for Unseen
# import os
# os.system('''CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4  python craa.py --task_mode zsl \
# --manualSeed 806 --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --ngh 2048 --ndh 2048 --lr_b 0.0002 --lr_i 0.0002 --lambda1 10 --critic_iter 5 --dataset FLO \
# --nclass_all 102 --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 \
# --seen_class_num 82 --base_class_num 18 --task_num 5 \
# --center_margin 200 --incenter_weight 0.8 \
# --protoSize 1024 --hSize 2048 --gamma 120 --epsilon 3 \
# --nepoch_base 90 --nepoch_incremental 50 --refine_fea \
# --syn_replay_num 100 --syn_previous_seen_num 100 --syn_unseen_num 30''')


## Incremental Learning for Seen
import os
os.system('''CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4  python craa.py --task_mode il \
--manualSeed 806 --preprocessing --cuda --image_embedding res101 --class_embedding att \
--ngh 2048 --ndh 2048 --lr_b 0.0002 --lr_i 0.0002 --lambda1 10 --critic_iter 5 --dataset FLO \
--nclass_all 102 --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 \
--seen_class_num 82 --base_class_num 18 --task_num 5 \
--center_margin 200 --incenter_weight 0.8 \
--protoSize 1024 --hSize 2048 --gamma 120 --epsilon 5 \
--nepoch_base 90 --nepoch_incremental 50 --refine_fea \
--syn_replay_num 100 --syn_previous_seen_num 100''')