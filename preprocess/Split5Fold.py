##########################################################################
## Multi Modal(Clinical, CT, PET) 데이터셋을 Croo-Validation하기 위해
## 5-fold 데이터셋 생성
#########################################################################
import os
import pandas as pd
import numpy as np
import random as rd

print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
print('■■ Generate 5-fold Data: Start')
df_clinical = pd.read_excel('../output/CLINICAL_2673_pre.xlsx')
np_pet_2673 = np.load('../output/50x128x128/PET_2673_50x128x128_normal.npy')
np_ct_2673 = np.load('../output/50x128x128/CT_2673_50x128x128_normal.npy')

seed_ = 12345
rd.seed(seed_)

alive = np.where(df_clinical['Deadstatus.event'] == 0)[0]
dead = np.where(df_clinical['Deadstatus.event'] == 1)[0]

rd.shuffle(alive)
rd.shuffle(dead)

p = [0, 0.2, 0.2, 0.2, 0.2, 0.2]
p = np.cumsum(np.array(p))

split_idx = []
for i in range(len(p)-1):
    split_num = np.hstack((alive[int(len(alive)*p[i]):int(len(alive)*p[i+1])], dead[int(len(dead)*p[i]):int(len(dead)*p[i+1])]))
    rd.shuffle(split_num)
    split_idx.append(split_num)

fold_1 = split_idx[0]
fold_2 = split_idx[1]
fold_3 = split_idx[2]
fold_4 = split_idx[3]
fold_5 = split_idx[4]

print('1-fold => {}'.format(len(fold_1)))
print('2-fold => {}'.format(len(fold_2)))
print('3-fold => {}'.format(len(fold_3)))
print('4-fold => {}'.format(len(fold_4)))
print('5-fold => {}'.format(len(fold_5)))

df_fold_1 = df_clinical.loc[fold_1]
df_fold_2 = df_clinical.loc[fold_2]
df_fold_3 = df_clinical.loc[fold_3]
df_fold_4 = df_clinical.loc[fold_4]
df_fold_5 = df_clinical.loc[fold_5]

output_path = '../output/5fold/50x128x128'

if not os.path.isdir(output_path):
    os.makedirs(output_path)

np.save('{}/CT_FOLD_1_{}'.format(output_path, len(fold_1)), np_ct_2673[split_idx[0]])
np.save('{}/CT_FOLD_2_{}'.format(output_path, len(fold_2)), np_ct_2673[split_idx[1]])
np.save('{}/CT_FOLD_3_{}'.format(output_path, len(fold_3)), np_ct_2673[split_idx[2]])
np.save('{}/CT_FOLD_4_{}'.format(output_path, len(fold_4)), np_ct_2673[split_idx[3]])
np.save('{}/CT_FOLD_5_{}'.format(output_path, len(fold_5)), np_ct_2673[split_idx[4]])

np.save('{}/PET_FOLD_1_{}'.format(output_path, len(fold_1)), np_pet_2673[split_idx[0]])
np.save('{}/PET_FOLD_2_{}'.format(output_path, len(fold_2)), np_pet_2673[split_idx[1]])
np.save('{}/PET_FOLD_3_{}'.format(output_path, len(fold_3)), np_pet_2673[split_idx[2]])
np.save('{}/PET_FOLD_4_{}'.format(output_path, len(fold_4)), np_pet_2673[split_idx[3]])
np.save('{}/PET_FOLD_5_{}'.format(output_path, len(fold_5)), np_pet_2673[split_idx[4]])


