import os
import time
import math
import pandas as pd
from Configuration import Config
from preprocess.Gen3dImage import Gen3dImage
from preprocess.Normalization3dImage import Normalization3dImage
from preprocess.PreprocessClinical import PreprocessClinical
from preprocess.SplitTrainTestSet import split_censored
import matplotlib.pyplot as plt
import seaborn as sns


#################
# 1.Data Load #
#################

clinical_df = pd.read_excel('./b_data/CPC_3776_v1.xlsx')
print(len(clinical_df))
dir_ct = '../b_data/LC_NSCLC_CT_n=3776/'  # Directory
dir_pet = '../b_data/LC_NSCLC_PET_n=3776/'  # Directory

# df_clinical = dr_clinical
ct_list = os.listdir(dir_ct)
pet_list = os.listdir(dir_pet)

print('■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■')
print('■□ 1.Number of Patients')
print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
print('■■ [CLINICAL Number of patients] → {}order'.format(len(clinical_df)))
print('■■ [CT Number of patients] → {}order'.format(len(ct_list)))
print('■■ [PET Number of patients] → {}order'.format(len(pet_list)))

# Warning Message Output
if not (len(clinical_df) == len(ct_list) == len(pet_list)):
    print('>> [MESSAGE] ERROR: 멀티모달데이터셋 환자수가 같지 않습니다.')
# ※ 처리 시간 계산
start = time.time()

#####################
# 2.PrePreprocess ##
#####################

# 2-1. Clinical
ppClinical = PreprocessClinical()
clinical_1260 = ppClinical.pre_clinical(clinical_df)

# 2-2. 3D이미지 생성 및 reshape
# ex: 50x128x128x1 → dim x width x height x channel

print('■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■')
print('■□ 2.Generate 3D Image')
gen3DImage = Gen3dImage()

# 환자 목록 생성
patient_list = list(clinical_df['PatientID'])  # 임상데이터셋에서 환자ID 목록 추출
print(f"patient list {type(patient_list)}")

# DICOM(CT, PET) 이미지 전처리
pixel_size = Config.get_pixel_szie()  # → IMAGE WIDTH & HEIGHT
dim = Config.get_dim()  # → IMAGE DIMENSION

config = (pixel_size, dim)
gen3DImage.gen_3d_Image('CT', dir_ct, patient_list)
gen3DImage.gen_3d_Image('PET', dir_pet, patient_list)


####################
# 3.Normalization #
####################

print('■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■')
print('■□ 3.Normalization 3D Image')
n3DImage = Normalization3dImage()

# 0~255 수치값으로 정규화
n3DImage.normal_3d_image('CT')
n3DImage.normal_3d_image('PET')

##########################
# 4.Train/Test 셋 분할 #
##########################

# TRAIN, TEST → 7:3 비율로 Split
# : Censored 데이터 고려해서 (0, 1)에서 7:3 비율로 층화추출
split_censored()

# ※ 처리시간 계산하기
print('>> [전처리 WORKING TIME] → {} sec'.format(math.floor(time.time() - start)))  # 현재시간 - 시작시간 = 실행시간
