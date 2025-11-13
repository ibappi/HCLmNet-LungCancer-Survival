######################################################################################
# Date: 2020.11.05
# Developer by 최철웅, 오승원
# Last updated Date: 2024.02.21
# Updated by Bappi
# Desc: DICOM(CT, PET) 3D 이미지를 0~255값으로 정규화 후 저장
#####################################################################################

import os
import numpy as np
from Configuration import Config


class Normalization3dImage:

    def dicom_normalization(self, imgs):
        # print('정규화 전: ', imgs.min(), '~', imgs.max())
        abs_imgs = abs(imgs[np.arange(imgs.shape[0])])
        normal_imgs = ((abs_imgs - abs_imgs.min()) / (abs_imgs.max() - abs_imgs.min())) * 255.
        # print('정규화 후: ', normal_imgs.min(), '~', normal_imgs.max())
        return (normal_imgs)

    def normal_3d_image(self, img_type):
        pixel_size = Config.get_pixel_szie()  # → IMAGE WIDTH & HEIGHT
        dim = Config.get_dim()  # → IMAGE DIMENSION

        print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
        print('■■ Normalization: Start')
        print('image type: ', img_type)
        np_imgs = np.load('../output/{}/{}_3776_{}.npy'.format(Config.img_shape, img_type, Config.img_shape))
        result = self.dicom_normalization(np_imgs)
        normal_imgs = np.array(result, dtype=np.float32)

        # 3D Numpy 이미지 저장하기
        output_path = '../output/{}'.format(Config.img_shape)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        np.save('../output/{}/{}_{}_{}_normal'.format(Config.img_shape, img_type, len(normal_imgs),
                                                      Config.img_shape), normal_imgs)
        print('>> [MESSAGE] SUCCESS: {} → 정규화 된 3D이미지(Numpy) 생성'.format(img_type))

        #  If multiple folder maintain for data

        # df_files = ['528', '731', '650']
        # for file in df_files:
        #     np_imgs = np.load('../output1/{}/{}_{}_{}.npy'.format(Config.img_shape, img_type, file, Config.img_shape))
        #     result = self.dicom_normalization(np_imgs)
        #     normal_imgs = np.array(result, dtype=np.float32)
        #
        #     # 3D Numpy 이미지 저장하기
        #     output_path = '../output1/{}'.format(Config.img_shape)
        #     if not os.path.isdir(output_path):
        #         os.makedirs(output_path)
        #
        #     np.save(
        #         '../output1/{}/{}_{}_{}_normal'.format(Config.img_shape, img_type, len(normal_imgs), Config.img_shape),
        #         normal_imgs)
        #     print('>> [MESSAGE] SUCCESS: {} → 정규화 된 3D이미지(Numpy) 생성'.format(img_type))
