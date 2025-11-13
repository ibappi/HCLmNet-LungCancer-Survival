import os
import numpy as np
import pandas as pd
import numpy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import cv2
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt

print("Loading clinical data...")
clinical_df = pd.read_excel('./b_data/CPC_3776_v1.xlsx')
print(clinical_df.isnull().sum())
print("Handling missing values...")
clinical_df.replace('.', np.nan, inplace=True)
clinical_df.fillna(method='ffill', inplace=True)  # Forward fill for simplicity, adjust as necessary
clinical_df2 = clinical_df.drop(columns=['histology', 'Smoking.status', 'Smoking.amount'], errors='ignore')

print(clinical_df)

print(clinical_df2.isnull().sum())
print(clinical_df2.describe())

print(clinical_df2)

print("Loading and processing CT/PET images...")
ct_images = './data/LC_NSCLC_CT_n=235/'
pet_images_ = './data/LC_NSCLC_PET_n=235/'

# # Select only numeric columns for correlation
# clinical_df_numeric = clinical_df2.select_dtypes(include=[np.number])
#
# clinical_df_numeric_with_id = clinical_df2[['PatientID']].join(clinical_df_numeric)
#
#
# # Function to calculate Z-scores and identify outliers for each row (patient)
# def calculate_row_outliers(df):
#     outlier_summary = {}
#     outliers_per_patient = {}
#
#     for column in df.columns:
#         if column == 'PatientID':
#             continue  # Skip the PatientID column for Z-score calculation
#
#         # Calculate Z-scores for each column
#         z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
#
#         # Identify outliers (Z-score > 3)
#         outliers = z_scores > 3
#
#         # Save outliers by row (patient) for this column
#         for index, is_outlier in outliers.items():
#             if is_outlier:
#                 patient_id = df.loc[index, 'PatientID']
#                 if patient_id not in outliers_per_patient:
#                     outliers_per_patient[patient_id] = []
#                 outliers_per_patient[patient_id].append(column)
#
#     return outliers_per_patient
#
#
# # Calculate outlier status for each patient
# patient_outliers = calculate_row_outliers(clinical_df_numeric_with_id)
#
# # Display outlier information for each patient (by PatientID)
# for patient_id, outlier_columns in patient_outliers.items():
#     print(f"Patient {patient_id} has outliers in columns: {outlier_columns}")
# clinical_df_numeric_with_id = clinical_df2[['PatientID']].join(clinical_df_numeric)

# Calculating correlation
# corr = clinical_df_numeric.corr()

# Plotting the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.show()

# Load the numpy and check the
data1 = np.load('./b_data/output/ct_images_train.npy')
print(data1[:1])
print("Shape of the ct data:", data1.shape)
data = np.load('./b_data/output/pet_images_train.npy')
print(data[:1])
print("Shape of the pet data:", data.shape)

case_index = 1  # Index for the patient or scan (0 to 126)
slice_index = 9  # Index for the depth (0 to 49)

# Extract the desired 2D slice; data[case_index, 0, slice_index] will give you (128, 128)
image_slice = data1[case_index, 0, slice_index]

# Check the shape before displaying
print("Shape of the selected slice:", image_slice.shape)

# Display the slice
plt.imshow(image_slice, cmap='gray')
plt.title(f'Patient {case_index}, Slice {slice_index}')
plt.axis('off')  # Hide axes for better visualization
plt.show()