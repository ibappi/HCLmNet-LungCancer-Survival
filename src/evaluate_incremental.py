# evaluate_incremental.py

import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from lifelines.utils import concordance_index
from TrainModel import MultimodalModel, device

# File paths
output_path_original = './b_data/output'
output_path_new = './s_data/new_output'

# Load original test data
X_clinical_val_orig = pd.read_excel(os.path.join(output_path_original, 'X_clinical_val.xlsx'))
y_clinical_val_orig = pd.read_excel(os.path.join(output_path_original, 'y_clinical_val.xlsx'))
ct_images_val_orig = np.load(os.path.join(output_path_original, 'ct_images_val.npy'))
pet_images_val_orig = np.load(os.path.join(output_path_original, 'pet_images_val.npy'))

# Load new test data
X_clinical_val_new = pd.read_excel(os.path.join(output_path_new, 'X_clinical_val.xlsx'))
y_clinical_val_new = pd.read_excel(os.path.join(output_path_new, 'y_clinical_val.xlsx'))
ct_images_val_new = np.load(os.path.join(output_path_new, 'ct_images_test.npy'))
pet_images_val_new = np.load(os.path.join(output_path_new, 'pet_images_test.npy'))

# Convert data to tensors (original test set)
X_val_tensor_orig = torch.tensor(X_clinical_val_orig.drop(columns=['PatientID']).values, dtype=torch.float32).to(device)
y_val_tensor_orig = torch.tensor(y_clinical_val_orig[['Survival.time', 'Deadstatus.event']].values,
                                 dtype=torch.float32).to(device)
ct_tensor_val_orig = torch.tensor(ct_images_val_orig, dtype=torch.float32).to(device)
pet_tensor_val_orig = torch.tensor(pet_images_val_orig, dtype=torch.float32).to(device)

# Convert data to tensors (new test set)
X_val_tensor_new = torch.tensor(X_clinical_val_new.drop(columns=['PatientID']).values, dtype=torch.float32).to(device)
y_val_tensor_new = torch.tensor(y_clinical_val_new[['Survival.time', 'Deadstatus.event']].values,
                                dtype=torch.float32).to(device)
ct_tensor_val_new = torch.tensor(ct_images_val_new, dtype=torch.float32).to(device)
pet_tensor_val_new = torch.tensor(pet_images_val_new, dtype=torch.float32).to(device)

# Load initial model (baseline model)
initial_model = MultimodalModel().to(device)
initial_model.load_state_dict(torch.load('initial_model.pth')['model_state_dict'])
initial_model.eval()

# Load updated model (after incremental learning)
updated_model = MultimodalModel().to(device)
updated_model.load_state_dict(torch.load('incremental_model.pth')['model_state_dict'])
updated_model.eval()


# # Evaluation Function
# def evaluate_model(model, X_tensor, ct_tensor, pet_tensor, y_tensor):
#     with torch.no_grad():
#         output = model(X_tensor, ct_tensor, pet_tensor)
#         survival_time, event_status = y_tensor[:, 0], y_tensor[:, 1]
#
#         # Calculate metrics
#         c_index = concordance_index(survival_time.cpu(), -output.cpu())
#         mae = mean_absolute_error(survival_time.cpu().numpy(), output.cpu().numpy())
#
#         return c_index, mae
#
#
# # Step 1: Baseline Performance on Original Test Set
# baseline_cindex, baseline_mae = evaluate_model(initial_model, X_val_tensor_orig, ct_tensor_val_orig,
#                                                pet_tensor_val_orig, y_val_tensor_orig)
#
# # Step 2: New Data Performance on New Test Set (Using Updated Model)
# new_data_cindex, new_data_mae = evaluate_model(updated_model, X_val_tensor_new, ct_tensor_val_new, pet_tensor_val_new,
#                                                y_val_tensor_new)
#
# # Step 3: Knowledge Retention on Original Test Set (Using Updated Model)
# retention_cindex, retention_mae = evaluate_model(updated_model, X_val_tensor_orig, ct_tensor_val_orig,
#                                                  pet_tensor_val_orig, y_val_tensor_orig)
#
# # Step 4: Forgetting (Difference between Baseline and Retention Performance)
# forgetting_cindex = baseline_cindex - retention_cindex
# forgetting_mae = retention_mae - baseline_mae
#
# # Define Scores
# baseline_score = (baseline_cindex - baseline_mae)  # Higher is better
# new_data_score = (new_data_cindex - new_data_mae)  # Higher is better
# retention_score = (retention_cindex - retention_mae)  # Higher is better
# forgetting_score = (forgetting_cindex + forgetting_mae)  # Lower is better (use negative)
#
# # Print all Scores
# print("\n=== Performance Summary ===")
# print(f"Baseline Performance Score (Original Data): {baseline_score:.4f}")
# print(f"New Data Performance Score (New Data): {new_data_score:.4f}")
# print(f"Knowledge Retention Score (Original Data): {retention_score:.4f}")
# print(f"Forgetting Score (Original Data): {forgetting_score:.4f}")
#
# # Define Scores
# baseline_score = (baseline_cindex - baseline_mae)  # Higher is better
# new_data_score = (new_data_cindex - new_data_mae)  # Higher is better
# retention_score = (retention_cindex - retention_mae)  # Higher is better
# forgetting_score = (forgetting_cindex + forgetting_mae)  # Lower is better

# Evaluation function (clearly defined and explained)
def evaluate_model(model, X_tensor, ct_tensor, pet_tensor, y_tensor):
    with torch.no_grad():
        output = model(X_tensor, ct_tensor, pet_tensor).squeeze()

        # Predicted survival times should directly represent days
        predicted_survival_days = output.cpu().numpy()
        actual_survival_days = y_tensor[:, 0].cpu().numpy()
        event_status = y_tensor[:, 1].cpu().numpy()

        # Concordance Index (Higher is better, 0~1)
        c_index = concordance_index(actual_survival_days, -predicted_survival_days, event_status)

        # Mean Absolute Error (Lower is better, measured in days)
        mae = mean_absolute_error(actual_survival_days, predicted_survival_days)

        return c_index, mae

# Calculate metrics separately
baseline_cindex, baseline_mae = evaluate_model(initial_model, X_val_tensor_orig, ct_tensor_val_orig, pet_tensor_val_orig, y_val_tensor_orig)

new_data_cindex, new_data_mae = evaluate_model(updated_model, X_val_tensor_new, ct_tensor_val_new, pet_tensor_val_new, y_val_tensor_new)

retention_cindex, retention_mae = evaluate_model(updated_model, X_val_tensor_orig, ct_tensor_val_orig, pet_tensor_val_orig, y_val_tensor_orig)

forgetting_cindex = baseline_cindex - retention_cindex
forgetting_mae = retention_mae - baseline_mae

# Print clear evaluation summary
print("\n=== Corrected Performance Metrics ===")

print(f"\nBaseline Performance (Original Data):")
print(f"  - Concordance Index: {baseline_cindex:.4f}")
print(f"  - MAE: {baseline_mae:.2f} days")

print(f"\nNew Data Performance (Incremental Data):")
print(f"  - Concordance Index: {new_data_cindex:.4f}")
print(f"  - MAE: {new_data_mae:.2f} days")

print(f"\nRetention Performance (Original Data after Incremental Learning):")
print(f"  - Concordance Index: {retention_cindex:.4f}")
print(f"  - MAE: {retention_mae:.2f} days")

print(f"\nForgetting (Performance Degradation):")
print(f"  - Concordance Index Decrease: {forgetting_cindex:.4f}")
print(f"  - MAE Increase: {forgetting_mae:.2f} days")
