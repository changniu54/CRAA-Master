#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
our CI-ZSL method
@author: Chang Niu
"""


# import os
# os.system('''CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4  python craa.py --task_mode gzsl \
# --manualSeed 4115 --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --ngh 2048 --ndh 2048 --lr_b 0.0001 --lr_i 0.0001 --lambda1 10 --critic_iter 5 --dataset SUN \
# --nclass_all 717 --batch_size 64 --nz 102 --attSize 102 --resSize 2048 \
# --seen_class_num 645 --base_class_num 129 --task_num 5 \
# --center_margin 120 --incenter_weight 0.8 \
# --protoSize 1024 --hSize 2048 --gamma 30 --epsilon 5 \
# --nepoch_base 200 --nepoch_incremental 100 --refine_fea \
# --syn_replay_num 20 --syn_previous_seen_num 20 --syn_unseen_num 10''')

# import os
# os.system('''CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4  python craa.py --task_mode zsl \
# --manualSeed 4115 --preprocessing --cuda --image_embedding res101 --class_embedding att \
# --ngh 2048 --ndh 2048 --lr_b 0.0001 --lr_i 0.0001 --lambda1 10 --critic_iter 5 --dataset SUN \
# --nclass_all 717 --batch_size 64 --nz 102 --attSize 102 --resSize 2048 \
# --seen_class_num 645 --base_class_num 129 --task_num 5 \
# --center_margin 120 --incenter_weight 0.8 \
# --protoSize 1024 --hSize 2048 --gamma 25 --epsilon 3 \
# --nepoch_base 40 --nepoch_incremental 40 --refine_fea \
# --syn_replay_num 20 --syn_previous_seen_num 20 --syn_unseen_num 50''')


import os
os.system('''CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4  python craa.py --task_mode il \
--manualSeed 4115 --preprocessing --cuda --image_embedding res101 --class_embedding att \
--ngh 2048 --ndh 2048 --lr_b 0.0001 --lr_i 0.0001 --lambda1 10 --critic_iter 5 --dataset SUN \
--nclass_all 717 --batch_size 64 --nz 102 --attSize 102 --resSize 2048 \
--seen_class_num 645 --base_class_num 129 --task_num 5 \
--center_margin 120 --incenter_weight 0.8 \
--protoSize 1024 --hSize 2048 --gamma 25 --epsilon 5 \
--nepoch_base 200 --nepoch_incremental 100 --refine_fea \
--syn_replay_num 20 --syn_previous_seen_num 20''')