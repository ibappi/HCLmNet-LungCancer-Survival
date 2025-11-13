# evaluate.py

import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from lifelines.utils import concordance_index
from TrainModel import MultimodalModel, device

# Load test data
output_path = './b_data/output'
X_clinical_val = pd.read_excel(os.path.join(output_path, 'X_clinical_val.xlsx'))
y_clinical_val = pd.read_excel(os.path.join(output_path, 'y_clinical_val.xlsx'))

# Prepare clinical data tensors
X_val_tensor = torch.tensor(X_clinical_val.drop(columns=['PatientID']).values, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_clinical_val[['Survival.time', 'Deadstatus.event']].values, dtype=torch.float32).to(
    device)

# Load CT and PET data
ct_images_val = np.load(os.path.join(output_path, 'ct_images_val.npy'))
pet_images_val = np.load(os.path.join(output_path, 'pet_images_val.npy'))

# Prepare CT and PET data tensors
ct_tensor_val = torch.tensor(ct_images_val, dtype=torch.float32).to(device)
pet_tensor_val = torch.tensor(pet_images_val, dtype=torch.float32).to(device)

# Load saved model
model = MultimodalModel().to(device)
model.load_state_dict(torch.load('initial_model.pth')['model_state_dict'])
model.eval()

with torch.no_grad():
    # Make predictions
    output = model(X_val_tensor, ct_tensor_val, pet_tensor_val)

    # Extract actual survival time and event status
    survival_time, event_status = y_val_tensor[:, 0], y_val_tensor[:, 1]

    # Calculate Concordance Index
    concordance = concordance_index(survival_time.cpu(), -output.cpu())
    print(f'Concordance Index: {concordance:.4f}')

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(survival_time.cpu().numpy(), output.cpu().numpy())
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
