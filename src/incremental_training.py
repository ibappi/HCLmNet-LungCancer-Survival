import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from TrainModel import MultimodalModel, device
from replay_memory import ReplayMemoryBuffer
from ewc import EWC

# Initialize Replay Memory
memory_capacity = 500  # Define memory capacity
memory_buffer = ReplayMemoryBuffer(capacity=memory_capacity)

# Paths to new training data
output_path_new = './s_data/new_output'
X_clinical_train_new = pd.read_excel(os.path.join(output_path_new, 'X_clinical_train.xlsx'))
y_clinical_train_new = pd.read_excel(os.path.join(output_path_new, 'y_clinical_train.xlsx'))

ct_images_train_new = np.load(os.path.join(output_path_new, 'ct_images_train.npy'))
pet_images_train_new = np.load(os.path.join(output_path_new, 'pet_images_train.npy'))

# Prepare tensors
X_train_tensor_new = torch.tensor(X_clinical_train_new.drop(columns=['PatientID']).values, dtype=torch.float32).to(
    device)
y_train_tensor_new = torch.tensor(y_clinical_train_new[['Survival.time', 'Deadstatus.event']].values,
                                  dtype=torch.float32).to(device)

ct_tensor_train_new = torch.tensor(ct_images_train_new, dtype=torch.float32).to(device)
pet_tensor_train_new = torch.tensor(pet_images_train_new, dtype=torch.float32).to(device)

# Load the initial model
model = MultimodalModel().to(device)
model.load_state_dict(torch.load('initial_model.pth')['model_state_dict'])

# Initialize EWC
ewc = EWC(model)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Populate replay memory buffer with old data (assuming you have access to it)
old_data_path = './s_data/old_output'
if os.path.exists(old_data_path):
    X_clinical_train_old = pd.read_excel(os.path.join(old_data_path, 'X_clinical_train.xlsx'))
    y_clinical_train_old = pd.read_excel(os.path.join(old_data_path, 'y_clinical_train.xlsx'))
    ct_images_train_old = np.load(os.path.join(old_data_path, 'ct_images_train.npy'))
    pet_images_train_old = np.load(os.path.join(old_data_path, 'pet_images_train.npy'))

    X_train_tensor_old = torch.tensor(X_clinical_train_old.drop(columns=['PatientID']).values, dtype=torch.float32).to(
        device)
    y_train_tensor_old = torch.tensor(y_clinical_train_old[['Survival.time', 'Deadstatus.event']].values,
                                      dtype=torch.float32).to(device)
    ct_tensor_train_old = torch.tensor(ct_images_train_old, dtype=torch.float32).to(device)
    pet_tensor_train_old = torch.tensor(pet_images_train_old, dtype=torch.float32).to(device)

    for i in range(len(X_train_tensor_old)):
        memory_buffer.add(X_train_tensor_old[i], ct_tensor_train_old[i], pet_tensor_train_old[i],
                          y_train_tensor_old[i])

# Calculate Fisher Information for EWC
if not memory_buffer.is_empty():
    ewc.calculate_fisher([(X_train_tensor_old[i], y_train_tensor_old[i]) for i in range(len(X_train_tensor_old))],
                         device=device)

# Fine-tune model on new data
model.train()
num_epochs = 10
batch_size = 32  # Define batch size for training

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(X_train_tensor_new), batch_size):
        # Prepare new batch
        batch_X_new = X_train_tensor_new[i:i + batch_size]
        batch_y_new = y_train_tensor_new[i:i + batch_size]
        batch_ct_new = ct_tensor_train_new[i:i + batch_size]
        batch_pet_new = pet_tensor_train_new[i:i + batch_size]

        # Sample from replay memory for ER and EICR
        replay_samples = memory_buffer.sample(batch_size // 2)
        replay_X, replay_ct, replay_pet, replay_y = zip(*replay_samples)

        replay_X = torch.stack(replay_X).to(device)
        replay_ct = torch.stack(replay_ct).to(device)
        replay_pet = torch.stack(replay_pet).to(device)
        replay_y = torch.stack(replay_y).to(device)

        # Combine new and replay data
        combined_X = torch.cat([batch_X_new, replay_X], dim=0)
        combined_ct = torch.cat([batch_ct_new, replay_ct], dim=0)
        combined_pet = torch.cat([batch_pet_new, replay_pet], dim=0)
        combined_y = torch.cat([batch_y_new, replay_y], dim=0)

        optimizer.zero_grad()

        # Forward pass
        output = model(combined_X, combined_ct, combined_pet)
        survival_time = combined_y[:, 0]

        # Loss calculation (MSE + EWC penalty)
        loss = criterion(output, survival_time) + ewc.penalty(model)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    # Update EWC parameters after every epoch
    ewc.update(model)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(X_train_tensor_new):.4f}")

# Save the updated model
torch.save({'model_state_dict': model.state_dict()}, 'incremental_model.pth')
print("Incremental model saved as 'incremental_model.pth'")
