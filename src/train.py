# train.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from pycox.models.loss import CoxPHLoss
from replay_memory import ReplayMemoryBuffer
from TrainModel import MultimodalModel, save_checkpoint, device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Hyperparameters
batch_size = 16
num_epochs = 20
replay_batch_size = 8
memory_capacity = 500

# Load your preprocessed data (Modify paths as needed)
output_path = '../b_data/output'
X_clinical_train = pd.read_excel(os.path.join(output_path, 'X_clinical_train.xlsx'))
y_clinical_train = pd.read_excel(os.path.join(output_path, 'y_clinical_train.xlsx'))
X_clinical_val = pd.read_excel(os.path.join(output_path, 'X_clinical_val.xlsx'))
y_clinical_val = pd.read_excel(os.path.join(output_path, 'y_clinical_val.xlsx'))

X_clinical_train = X_clinical_train[:2000]
y_clinical_train = y_clinical_train[:2000]
X_clinical_val = X_clinical_val[:300]
y_clinical_val = y_clinical_val[:300]

# Convert clinical data to tensors
X_clinical_train = X_clinical_train.drop(columns=['PatientID'])
X_clinical_train = X_clinical_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_train_tensor = torch.tensor(X_clinical_train.values, dtype=torch.float16).to(device)
y_train_tensor = torch.tensor(y_clinical_train[['Survival.time', 'Deadstatus.event']].values, dtype=torch.float16).to(device)

# Load CT and PET data (assume already preprocessed)
ct_images_train = np.load(os.path.join(output_path, 'ct_images_train.npy'))
pet_images_train = np.load(os.path.join(output_path, 'pet_images_train.npy'))
ct_images_val = np.load(os.path.join(output_path, 'ct_images_val.npy'))
pet_images_val = np.load(os.path.join(output_path, 'pet_images_val.npy'))

ct_images_train = ct_images_train[:2000]
pet_images_train = pet_images_train[:2000]
ct_images_val = ct_images_val[:300]
pet_images_val = pet_images_val[:300]

torch.cuda.empty_cache()

# Ensure the tensors are of the correct type and dimensions (float32, not float16)
ct_tensor_train = torch.tensor(ct_images_train, dtype=torch.float16).to(device)
pet_tensor_train = torch.tensor(pet_images_train, dtype=torch.float16).to(device)
ct_tensor_val = torch.tensor(ct_images_val, dtype=torch.float16).to(device)
pet_tensor_val = torch.tensor(pet_images_val, dtype=torch.float16).to(device)

# Create TensorDataset
train_dataset = TensorDataset(X_train_tensor, ct_tensor_train, pet_tensor_train, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create validation dataset
X_clinical_val = X_clinical_val.drop(columns=['PatientID'])
X_clinical_val = X_clinical_val.apply(pd.to_numeric, errors='coerce').fillna(0)
X_val_tensor = torch.tensor(X_clinical_val.values, dtype=torch.float16).to(device)
y_val_tensor = torch.tensor(y_clinical_val[['Survival.time', 'Deadstatus.event']].values, dtype=torch.float16).to(device)

valid_dataset = TensorDataset(X_val_tensor, ct_tensor_val, pet_tensor_val, y_val_tensor)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
model = MultimodalModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
criterion = CoxPHLoss()
scaler = GradScaler()

# Initialize replay buffer
replay_buffer = ReplayMemoryBuffer(memory_capacity)

# Training loop with replay buffer
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for clinical_data, ct_data, pet_data, target in train_loader:
        optimizer.zero_grad()

        # Store in replay buffer
        replay_buffer.add(clinical_data, ct_data, pet_data, target)

        with autocast():
            # Model forward pass
            output = model(clinical_data, ct_data, pet_data)
            survival_time, event_status = target[:, 0], target[:, 1]
            loss = criterion(output, survival_time, event_status)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Sample from replay buffer
        if replay_buffer.is_full():
            replay_samples = replay_buffer.sample(replay_batch_size)
            for clinical_data, ct_data, pet_data, target in replay_samples:
                output = model(clinical_data, ct_data, pet_data)
                survival_time, event_status = target[:, 0], target[:, 1]
                loss = criterion(output, survival_time, event_status)
                loss.backward()
                optimizer.step()

    train_losses.append(running_loss / len(train_loader))
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}')

    # Validation step
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to compute gradients for validation
        for clinical_data, ct_data, pet_data, target in valid_loader:
            output = model(clinical_data, ct_data, pet_data)
            survival_time, event_status = target[:, 0], target[:, 1]
            loss = criterion(output, survival_time, event_status)
            val_loss += loss.item()

    val_losses.append(val_loss / len(valid_loader))
    print(f'Validation Loss: {val_losses[-1]:.4f}')

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        save_checkpoint(epoch, model, optimizer, path=f'checkpoint_epoch_{epoch + 1}.pth')

print("Training completed.")
torch.save({'model_state_dict': model.state_dict()}, 'initial_model.pth')
print("Model saved as 'initial_model.pth'")
