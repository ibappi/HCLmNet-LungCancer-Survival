######################################################################################
# Date: 2020.11.08
# Last updated Date: 2024.02.21
# Updated by Bappi
# Developer by 최철웅, 오승원
# Desc: Clinical, CT, PET을 Train:TEST → 7:3 비율로 분할하고 저장하는 코드
#####################################################################################

import os
import numpy as np
import random as rd
import pandas as pd
from Configuration import Config


def split_censored():
    df_clinical_1260 = pd.read_excel('../output/CLINICAL_3776_pre.xlsx')
    df_clinical = df_clinical_1260

    # df_clinical_834 = pd.read_excel('../output1/CLINICAL_731_pre.xlsx')
    # df_clinical_750 = pd.read_excel('../output1/CLINICAL_650_pre.xlsx')
    # df_clinical = pd.concat([df_clinical_538, df_clinical_834, df_clinical_750], ignore_index=True)

    normal_ct_1260 = np.load('../output/{}/CT_3776_{}_normal.npy'.format(Config.img_shape, Config.img_shape))
    normal_ct = normal_ct_1260
    # normal_ct_834 = np.load('../output1/{}/CT_731_{}_normal.npy'.format(Config.img_shape, Config.img_shape))
    # normal_ct_750 = np.load('../output1/{}/CT_650_{}_normal.npy'.format(Config.img_shape, Config.img_shape))
    # normal_ct = np.concatenate((normal_ct_538, normal_ct_834, normal_ct_750), axis=0)

    normal_pet_1260 = np.load('../output/{}/PET_3776_{}_normal.npy'.format(Config.img_shape, Config.img_shape))
    normal_pet = normal_pet_1260
    # normal_pet_538 = np.load('../output1/{}/PET_528_{}_normal.npy'.format(Config.img_shape, Config.img_shape))
    # normal_pet_750 = np.load('../output1/{}/PET_650_{}_normal.npy'.format(Config.img_shape, Config.img_shape))
    # normal_pet = np.concatenate((normal_pet_538, normal_pet_834, normal_pet_750), axis=0)

    print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
    print('■■ Split TRAIN/TEST: Start')

    # TRAIN, TEST : 7, 3 비율로 자르는 Index Num 생성
    seed_ = 12345
    rd.seed(seed_)

    alive = np.where(df_clinical['Deadstatus.event'] == 0)[0]
    dead = np.where(df_clinical['Deadstatus.event'] == 1)[0]
    print('alive: ', len(alive))
    print('dead: ', len(dead))

    rd.shuffle(alive)
    rd.shuffle(dead)

    p = [0, 0.8, 0.1, 0.11]
    p = np.cumsum(np.array(p))

    split_idx = []
    for i in range(len(p) - 1):
        split_num = np.hstack((alive[int(len(alive) * p[i]):int(len(alive) * p[i + 1])],
                               dead[int(len(dead) * p[i]):int(len(dead) * p[i + 1])]))
        rd.shuffle(split_num)
        split_idx.append(split_num)

    train_num = split_idx[0]
    print('train', train_num)
    valid_num = split_idx[1]
    print('valid', valid_num)
    test_num = split_idx[2]
    print('test', test_num)

    print('■■ TRAIN({}) + VALID({}) + TEST({}) = {}'.format(len(train_num), len(test_num), len(valid_num),
                                                            (len(train_num) + len(valid_num) + len(test_num))))

    # CLINICAL 임상데이터 Split
    df_train = df_clinical.loc[train_num]
    df_valid = df_clinical.loc[valid_num]
    df_test = df_clinical.loc[test_num]

    df_train.to_excel('../output/TRAIN_CLINICAL_{}_pre.xlsx'.format(len(df_train)), index=False)
    df_valid.to_excel('../output/VALID_CLINICAL_{}_pre.xlsx'.format(len(df_valid)), index=False)
    df_test.to_excel('../output/TEST_CLINICAL_{}_pre.xlsx'.format(len(df_test)), index=False)

    # DICOM CT&PET 이미지 Split
    pixel_size = Config.get_pixel_szie()  # → IMAGE WIDTH & HEIGHT
    dim = Config.get_dim()  # → IMAGE DIMENSION
    train_ct = normal_ct[split_idx[0]]
    valid_ct = normal_ct[split_idx[1]]
    test_ct = normal_ct[split_idx[2]]

    train_pet = normal_pet[split_idx[0]]
    valid_pet = normal_pet[split_idx[1]]
    test_pet = normal_pet[split_idx[2]]

    output_path = '../output/{}'.format(Config.img_shape)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    np.save('../output/{}/TRAIN_CT_{}_{}_normal'.format(Config.img_shape, len(train_ct), Config.img_shape), train_ct)
    np.save('../output/{}/VALID_CT_{}_{}_normal'.format(Config.img_shape, len(valid_ct), Config.img_shape), valid_ct)
    np.save('../output/{}/TEST_CT_{}_{}_normal'.format(Config.img_shape, len(test_ct), Config.img_shape), test_ct)
    np.save('../output/{}/TRAIN_PET_{}_{}_normal'.format(Config.img_shape, len(train_pet), Config.img_shape),
            train_pet)
    np.save('../output/{}/VALID_PET_{}_{}_normal'.format(Config.img_shape, len(valid_pet), Config.img_shape),
            valid_pet)
    np.save('../output/{}/TEST_PET_{}_{}_normal'.format(Config.img_shape, len(test_pet), Config.img_shape), test_pet)

    return split_idx
