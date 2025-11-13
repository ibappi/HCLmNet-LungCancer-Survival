

class Config:
    # Data Shape(50x128x128)
    pixel_size = 128  # → IMAGE WIDTH x HEIGHT
    dim = 50         # → IMAGE DIMENSION
    img_shape = '{}x{}x{}'.format(dim, pixel_size, pixel_size) # '50x128x128'

    # CNN Model Config
    model_surv = 'DeepSurv'            # DeepSurv
    model_CNN = 'Resnet3D'             # RESNET3D
    model_depth = 18                   # RESNET3D Layers
    batch_size = 16                    # AI Model Batch Size
    epoch = 50                         # AI Model Epoch
    dir_save_machine = '../model/{}'.format(img_shape)   # → AI 모델 저장경로

    model_name = 'SP_CLINICAL+CT+PET_{}_{}{}_{}_model.h5'.format(img_shape, model_CNN, model_depth, model_surv)
    # ex) D:/ai_model/PET_CLINICAL_50x128x128_Resnet3D34_DeepSurv_model.h5

    # clinical_path = ''
    # ct_path = ''
    # pet_path = ''

    @staticmethod
    def get_dim():
        return Config.dim

    @staticmethod
    def get_pixel_szie():
        return Config.pixel_size

    @staticmethod
    def get_data_shape():
        return '{}x{}x{}'.format(Config.dim, Config.pixel_size, Config.pixel_size)