import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import cv2
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
import random
from tqdm import tqdm  # Progress bar for monitoring


# Configuration class
class Config:
    @staticmethod
    def get_pixel_size():
        return 128  # Image width and height

    @staticmethod
    def get_dim():
        return 160  # Image depth


# Set output path for processed data
output_path = './data/output'
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# Load clinical data
print("Loading clinical data...")
clinical_df = pd.read_excel('./data/filtered_clinical_and_DNA_235_final_fixed.xlsx')

# Handle missing or erroneous values in clinical data
print("Handling missing values...")
clinical_df.replace('.', np.nan, inplace=True)
clinical_df.fillna(method='ffill', inplace=True)  # Forward fill for simplicity, adjust as necessary

# Filter necessary columns
print("Filtering relevant columns in clinical data...")
# Dead = 1 and Alive = 0
clinical_columns = [
    'PatientID', 'gender', 'age', 'Survival.time', 'Deadstatus.event',
    'Overall.stage', 'Clinical.T.stage', 'Clinical.N.stage',
    'Clinical.M.stage', 'Smoking.status', 'Smoking.amount',
    'SNV_CNT', 'HETE_CNT', 'HOMO_CNT'
]
clinical_df = clinical_df[clinical_columns]
clinical_df2 = clinical_df.drop(columns=['histology', 'Smoking.status', 'Smoking.amount', 'FILE_NAME'], errors='ignore')

# Normalize DNA columns, age, and Smoking.amount using MinMaxScaler to bring values between 0-1
print("Normalizing DNA columns, age, and Smoking.amount to 0-1 range...")
scaler = MinMaxScaler()
clinical_df2[['SNV_CNT', 'HETE_CNT', 'HOMO_CNT', 'age']] = scaler.fit_transform(
    clinical_df2[['SNV_CNT', 'HETE_CNT', 'HOMO_CNT', 'age']]
)

# Encode categorical features
print("Encoding categorical variables...")
clinical_df2 = pd.get_dummies(clinical_df2, columns=['gender', 'Overall.stage',
                                                     'Clinical.T.stage',
                                                     'Clinical.N.stage',
                                                     'Clinical.M.stage'], drop_first=True)

# Split the clinical data
print("Splitting clinical data into train, validation, and test sets...")
X_clinical = clinical_df2.drop(columns=['Survival.time', 'Deadstatus.event'], errors='ignore')
y_clinical = clinical_df2[['Survival.time', 'Deadstatus.event']]

X_clinical_train, X_clinical_test, y_clinical_train, y_clinical_test = train_test_split(
    X_clinical, y_clinical, test_size=0.2, random_state=42, stratify=y_clinical['Deadstatus.event']
)

# Further split the test set into validation set
X_clinical_val, X_clinical_test, y_clinical_val, y_clinical_test = train_test_split(
    X_clinical_test, y_clinical_test, test_size=0.5, random_state=42, stratify=y_clinical_test['Deadstatus.event']
)

# Now the split should result in 188 train, 24 validation, and 23 test patients
print(
    f"Train set size: {len(X_clinical_train)}, Validation set size: {len(X_clinical_val)}, Test set size: {len(X_clinical_test)}")

# Save the clinical data into separate Excel files for train, val, and test
print("Saving processed clinical data into separate Excel files...")
X_clinical_train.to_excel(os.path.join(output_path, 'X_clinical_train.xlsx'), index=False)
y_clinical_train.to_excel(os.path.join(output_path, 'y_clinical_train.xlsx'), index=False)

X_clinical_val.to_excel(os.path.join(output_path, 'X_clinical_val.xlsx'), index=False)
y_clinical_val.to_excel(os.path.join(output_path, 'y_clinical_val.xlsx'), index=False)

X_clinical_test.to_excel(os.path.join(output_path, 'X_clinical_test.xlsx'), index=False)
y_clinical_test.to_excel(os.path.join(output_path, 'y_clinical_test.xlsx'), index=False)

print("Preprocessing complete and clinical data saved.")


# Augmentation function
def augment_image(img3d):
    """Applies augmentation to a 3D image (CT or PET)."""
    # Randomly apply transformations
    if random.random() < 0.5:
        img3d = np.flip(img3d, axis=1)  # Horizontal flip
    if random.random() < 0.5:
        img3d = np.flip(img3d, axis=2)  # Vertical flip
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)  # Small rotation
        for i in range(img3d.shape[1]):  # Apply to each slice
            img3d[0, i, :, :] = rotate_image(img3d[0, i, :, :], angle)

    # Add small random noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.01, img3d.shape)
        img3d = img3d + noise

    return img3d


def rotate_image(image, angle):
    """Rotates a 2D image by a given angle."""
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


# Step 1: Calculate global min and max values across the dataset
def calculate_global_min_max(data_dir, patient_list, img_type):
    min_val, max_val = np.inf, -np.inf
    for patient_id in patient_list:
        if img_type == 'CT':
            patient_id = 'CT_' + patient_id

        dcm_imgs = sorted(os.listdir(os.path.join(data_dir, patient_id)), key=lambda img_name: int(img_name[2:]))
        slices = [pydicom.dcmread(os.path.join(data_dir, patient_id, img)) for img in dcm_imgs]

        for s in slices:
            min_val = min(min_val, s.pixel_array.min())
            max_val = max(max_val, s.pixel_array.max())

    return min_val, max_val


# Step 2: Use the global min and max to normalize the dataset
def normalize_image(img3d, global_min, global_max):
    return (img3d - global_min) / (global_max - global_min)


# Load DICOM images with normalization and optional augmentation
def load_dicom_images(img_type, data_dir, patient_list, global_min, global_max, augment=False):
    pixel_size = Config.get_pixel_size()  # Assume you have a Config class that provides pixel size and dimensions
    dim = Config.get_dim()  # Assume you have a Config class that provides dimensions
    img3d_list = []

    print(f"Loading {img_type} images...")

    for patient_id in tqdm(patient_list, desc=f"Processing {img_type} images"):
        if img_type == 'CT':
            patient_id = 'CT_' + patient_id

        dcm_imgs = sorted(os.listdir(os.path.join(data_dir, patient_id)), key=lambda img_name: int(img_name[2:]))
        slices = [pydicom.dcmread(os.path.join(data_dir, patient_id, img)) for img in dcm_imgs]
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        img_shape = [pixel_size, pixel_size]
        img_shape.insert(0, len(slices))
        img3d = np.zeros(img_shape)

        for i, s in enumerate(slices):
            if slices[0].pixel_array.shape[0] != pixel_size:
                resized_img = cv2.resize(s.pixel_array, (pixel_size, pixel_size))
                img3d[i, :, :] = resized_img
            else:
                img3d[i, :, :] = s.pixel_array

        if dim > len(slices):
            background_val = -2000 if img_type == 'CT' else 0
            padding_img3d = np.full([(dim - len(slices)), pixel_size, pixel_size], background_val, dtype=np.float32)
            img3d = np.concatenate((img3d, padding_img3d), axis=0)
        else:
            resize_factor = [dim / len(slices), 1, 1]
            img3d = scipy.ndimage.zoom(img3d, resize_factor)

        img3d = np.reshape(img3d, (1, dim, pixel_size, pixel_size))

        # Apply dataset-wide normalization using global min and max values
        img3d = normalize_image(img3d, global_min, global_max)

        # Apply augmentation if enabled
        if augment:
            img3d = augment_image(img3d)

        img3d_list.append(img3d)

    return np.array(img3d_list)


start_time = time.time()
# Main Code for Loading and Processing CT and PET images
# --- PatientID used in CT/PET loading ---
train_patients = X_clinical_train['PatientID'].values
val_patients = X_clinical_val['PatientID'].values
test_patients = X_clinical_test['PatientID'].values

# Step 3: Compute global min and max values for CT and PET datasets
global_min_ct, global_max_ct = calculate_global_min_max('./data/LC_NSCLC_CT_n=235/', train_patients, 'CT')
global_min_pet, global_max_pet = calculate_global_min_max('./data/LC_NSCLC_PET_n=235/', train_patients, 'PET')

# Load and process CT and PET images with augmentation for the training set
print("Loading and processing CT images...")
ct_images_train = load_dicom_images('CT', './data/LC_NSCLC_CT_n=235/', train_patients, global_min_ct, global_max_ct,
                                    augment=True)
ct_images_val = load_dicom_images('CT', './data/LC_NSCLC_CT_n=235/', val_patients, global_min_ct, global_max_ct)
ct_images_test = load_dicom_images('CT', './data/LC_NSCLC_CT_n=235/', test_patients, global_min_ct, global_max_ct)

print("Loading and processing PET images...")
pet_images_train = load_dicom_images('PET', './data/LC_NSCLC_PET_n=235/', train_patients, global_min_pet,
                                     global_max_pet, augment=True)
pet_images_val = load_dicom_images('PET', './data/LC_NSCLC_PET_n=235/', val_patients, global_min_pet, global_max_pet)
pet_images_test = load_dicom_images('PET', './data/LC_NSCLC_PET_n=235/', test_patients, global_min_pet, global_max_pet)

print(f"Preprocessing and augmentation complete. Total time: {time.time() - start_time:.2f} seconds")

# Save processed CT and PET data
print("Saving processed CT and PET images...")
np.save(os.path.join(output_path, 'ct_images_train.npy'), ct_images_train)
np.save(os.path.join(output_path, 'pet_images_train.npy'), pet_images_train)
np.save(os.path.join(output_path, 'ct_images_val.npy'), ct_images_val)
np.save(os.path.join(output_path, 'pet_images_val.npy'), pet_images_val)
np.save(os.path.join(output_path, 'ct_images_test.npy'), ct_images_test)
np.save(os.path.join(output_path, 'pet_images_test.npy'), pet_images_test)

print("Preprocessing and augmentation complete.")
