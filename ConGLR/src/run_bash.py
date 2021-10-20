# -*- coding: utf-8 -*-

import os


def train():
    gpu = 3
    lr = 0.001  # 001
    l2 = 0.0005  # 0005
    epoch = 10
    # dataset_list = ['fb237','WN18RR','nell']
    # version = [1,2,3,4]

    dataset_list = ['nell']
    version = [4]
    for dataset in dataset_list:
        for v in version:
            temp_cmd = f'python train.py -d {dataset}_v{v} -e {dataset}_v{v} --gpu {gpu} --lr {lr} --l2 {l2} --num_epochs {epoch}'
            os.system(temp_cmd)

            # test
            temp_cmd = f'python test_auc.py -d {dataset}_v{v}_ind -e {dataset}_v{v} --gpu {gpu}'
            os.system(temp_cmd)

            temp_cmd = f'python test_auc.py -d {dataset}_v{v}_ind -e {dataset}_v{v} --gpu {gpu} --num_neg_samples_per_link {50}'
            os.system(temp_cmd)


def test():
    gpu = 0
    num_neg_samples_per_link = 50
    # dataset_list = ['fb237','WN18RR','nell']
    # version = [1,2,3,4]

    dataset_list = ['WN18RR']
    version = [1,2,3]
    for dataset in dataset_list:
        for v in version:
            temp_cmd = f'python test_auc.py -d {dataset}_v{v}_ind -e {dataset}_v{v} --gpu {gpu} --num_neg_samples_per_link {num_neg_samples_per_link}'
            os.system(temp_cmd)


train()
# test()