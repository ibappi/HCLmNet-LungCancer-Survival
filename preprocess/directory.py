import os
import shutil
import pandas as pd

# Load the clinical data with 235 IDs
clinical_235 = pd.read_excel('./data/filtered_clinical_and_DNA_1275_final_fixed.xlsx')

# Directories for CT and PET
dir_ct = './data/LC_NSCLC_CT_n=1300/'  # Source directory for CT
dir_pet = './data/LC_NSCLC_PET_n=1300/'  # Source directory for PET

# Target directories to move matched CT and PET folders
move_ct = './data/New_folder/LC_NSCLC_CT_n=1300/'  # Target directory for CT
move_pet = './data/New_folder/LC_NSCLC_PET_n=1300/'  # Target directory for PET

# Get the list of patient IDs from the clinical_235 file
ids_dna_235 = set(clinical_235['PatientID'])

# Get the list of patient IDs in the CT and PET directories
ct_ids = {folder_name.replace('CT_', '') for folder_name in os.listdir(dir_ct)}
pet_ids = set(os.listdir(dir_pet))

# Find the intersection of IDs between clinical data, CT, and PET directories
matched_ids = ids_dna_235 & ct_ids & pet_ids

# Filter clinical data to only keep rows with matched IDs
# clinical_matched = clinical_235[clinical_235['PatientID'].isin(matched_ids)]
#
# # Save the filtered clinical data to a new Excel file
# clinical_matched.to_excel('./data/CPC_1300_matched.xlsx', index=False)

#........ Ensure the target directories exist
os.makedirs(move_ct, exist_ok=True)
os.makedirs(move_pet, exist_ok=True)


# Function to move matched CT folders (removing "CT_" prefix from folder names)
def move_matched_ct_folders(source_dir, target_dir, matched_ids):
    for folder_name in os.listdir(source_dir):
        # Extract the patient ID by removing "CT_" prefix
        patient_id = folder_name.replace('CT_', '')

        if patient_id in matched_ids:
            source_folder = os.path.join(source_dir, folder_name)
            target_folder = os.path.join(target_dir, folder_name)

            # Move the folder to the new directory
            shutil.move(source_folder, target_folder)
            print(f"Moved CT folder {folder_name} to {target_dir}")


# Function to move matched PET folders (no prefix)
def move_matched_pet_folders(source_dir, target_dir, matched_ids):
    for folder_name in os.listdir(source_dir):
        # Folder name is the patient ID for PET data
        patient_id = folder_name

        if patient_id in matched_ids:
            source_folder = os.path.join(source_dir, folder_name)
            target_folder = os.path.join(target_dir, folder_name)

            # Move the folder to the new directory
            shutil.move(source_folder, target_folder)
            print(f"Moved PET folder {folder_name} to {target_dir}")


# Move matched CT folders
move_matched_ct_folders(dir_ct, move_ct, ids_dna_235)

# Move matched PET folders
move_matched_pet_folders(dir_pet, move_pet, ids_dna_235)

print("All matched CT and PET folders have been moved successfully.")
