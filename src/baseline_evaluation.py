# baseline_evaluation.py

import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from lifelines.utils import concordance_index
from TrainModel import MultimodalModel, device

# Paths to original test data
output_path_old = './data/output'
X_clinical_val_old = pd.read_excel(os.path.join(output_path_old, 'X_clinical_val.xlsx'))
y_clinical_val_old = pd.read_excel(os.path.join(output_path_old, 'y_clinical_val.xlsx'))

ct_images_val_old = np.load(os.path.join(output_path_old, 'ct_images_val.npy'))
pet_images_val_old = np.load(os.path.join(output_path_old, 'pet_images_val.npy'))

# Prepare tensors
X_val_tensor_old = torch.tensor(X_clinical_val_old.drop(columns=['PatientID']).values, dtype=torch.float32).to(device)
y_val_tensor_old = torch.tensor(y_clinical_val_old[['Survival.time', 'Deadstatus.event']].values, dtype=torch.float32).to(device)

ct_tensor_val_old = torch.tensor(ct_images_val_old, dtype=torch.float32).to(device)
pet_tensor_val_old = torch.tensor(pet_images_val_old, dtype=torch.float32).to(device)

# Load the saved model
model = MultimodalModel().to(device)
model.load_state_dict(torch.load('final_model.pth')['model_state_dict'])
model.eval()

# Evaluate Baseline Performance
with torch.no_grad():
    output_old = model(X_val_tensor_old, ct_tensor_val_old, pet_tensor_val_old)
    survival_time_old, event_status_old = y_val_tensor_old[:, 0], y_val_tensor_old[:, 1]

    # Concordance Index and MAE
    concordance_old = concordance_index(survival_time_old.cpu(), -output_old.cpu())
    mae_old = mean_absolute_error(survival_time_old.cpu().numpy(), output_old.cpu().numpy())
    print(f'Baseline Concordance Index: {concordance_old:.4f}')
    print(f'Baseline MAE: {mae_old:.4f}')
