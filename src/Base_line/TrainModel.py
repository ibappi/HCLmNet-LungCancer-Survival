import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from pycox.models.loss import CoxPHLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Swin Transformer components for 2D slices
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels * 160, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, depth, channels, height, width = x.shape
        x = x.view(batch_size, depth * channels, height, width)
        return self.proj(x).flatten(2).transpose(1, 2)


class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mlp_ratio=4.0):
        super(SwinTransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out  # Skip connection
        x = self.norm2(x)
        x = x + self.mlp(x)  # Skip connection
        return x


class SwinTransformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, num_heads=4, num_layers=4, window_size=7, patch_size=4):
        super(SwinTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.layers = nn.ModuleList([SwinTransformerBlock(embed_dim, num_heads, window_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.mean(dim=1)


class CrossAttention(nn.Module):
    def __init__(self, input_dim_lstm, input_dim_ct, input_dim_pet):
        super(CrossAttention, self).__init__()
        self.linear_q = nn.Linear(input_dim_lstm, input_dim_lstm)  # From LSTM
        self.linear_k = nn.Linear(input_dim_ct, input_dim_lstm)  # From CT
        self.linear_v = nn.Linear(input_dim_pet, input_dim_lstm)  # From PET

    def forward(self, query, key, value):
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        attention_weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
        output = torch.matmul(attention_weights, value)
        return output


class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.lstm = nn.LSTM(input_size=26, hidden_size=50, num_layers=2, batch_first=True, dropout=0.5)
        self.batch_norm_lstm = nn.BatchNorm1d(50)
        self.swin_transformer_ct = SwinTransformer()
        self.swin_transformer_pet = SwinTransformer()
        self.cross_attention = CrossAttention(input_dim_lstm=50, input_dim_ct=96, input_dim_pet=96)
        self.fc = nn.Sequential(
            nn.Linear(50 + 50, 50),  # Updated to 50 to match concatenated dimensions
            nn.Dropout(0.6),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, clinical_data, ct_data, pet_data):
        # print(f"Clinical data shape: {clinical_data.shape}")
        # print(f"CT data shape: {ct_data.shape}")
        # print(f"PET data shape: {pet_data.shape}")

        if clinical_data.dim() == 2:
            clinical_data = clinical_data.unsqueeze(1)

        lstm_out, _ = self.lstm(clinical_data)
        lstm_out = lstm_out[:, -1, :]
        # print(f"LSTM output shape: {lstm_out.shape}")
        lstm_out = self.batch_norm_lstm(lstm_out)
        # print(f"LSTM output shape after batchNor: {lstm_out.shape}")
        ct_out = self.swin_transformer_ct(ct_data)
        # print(f"CT transformer output shape: {ct_out.shape}")

        pet_out = self.swin_transformer_pet(pet_data)
        # print(f"PET transformer output shape: {pet_out.shape}")

        attention_out = self.cross_attention(lstm_out, ct_out, pet_out)
        # print(f"Cross attention output shape: {attention_out.shape}")

        combined = torch.cat((lstm_out, attention_out), dim=1)
        output = self.fc(combined)
        # print(f"Final output shape: {output.shape}")

        return output


# Define save and load functions
def save_checkpoint(epoch, model, optimizer, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
    else:
        print(f"No checkpoint found at {path}")


if __name__ == '__main__':
    # Load data paths
    output_path = './data/output'

    # Load clinical features and targets from separate Excel files
    X_clinical_train = pd.read_excel(os.path.join(output_path, 'X_clinical_train.xlsx'))
    y_clinical_train = pd.read_excel(os.path.join(output_path, 'y_clinical_train.xlsx'))
    X_clinical_val = pd.read_excel(os.path.join(output_path, 'X_clinical_val.xlsx'))
    y_clinical_val = pd.read_excel(os.path.join(output_path, 'y_clinical_val.xlsx'))

    # Convert to NumPy arrays
    X_clinical_train_cleaned = X_clinical_train.drop(columns=['PatientID']).select_dtypes(include=[np.number])
    y_clinical_train = y_clinical_train[['Survival.time', 'Deadstatus.event']].values.astype(np.float32)
    X_clinical_val_cleaned = X_clinical_val.drop(columns=['PatientID']).select_dtypes(include=[np.number])
    y_clinical_val = y_clinical_val[['Survival.time', 'Deadstatus.event']].values.astype(np.float32)
    X_clinical_train_cleaned = X_clinical_train_cleaned.values.astype(np.float32)
    X_clinical_val_cleaned = X_clinical_val_cleaned.values.astype(np.float32)

    # Convert clinical data to tensors
    X_train_tensor = torch.tensor(X_clinical_train_cleaned, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_clinical_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_clinical_val_cleaned, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_clinical_val, dtype=torch.float32).to(device)

    # Load CT and PET images
    ct_images_train = np.load(os.path.join(output_path, 'ct_images_train.npy'), allow_pickle=True)
    pet_images_train = np.load(os.path.join(output_path, 'pet_images_train.npy'), allow_pickle=True)
    ct_images_val = np.load(os.path.join(output_path, 'ct_images_val.npy'), allow_pickle=True)
    pet_images_val = np.load(os.path.join(output_path, 'pet_images_val.npy'), allow_pickle=True)

    # Flatten 3D CT and PET images into 2D slices and load them as tensors
    def flatten_3d_to_2d(image_array):
        num_slices = image_array.shape[1]
        return image_array.reshape(-1, 1, image_array.shape[2], image_array.shape[3])

    ct_images_train_2d = flatten_3d_to_2d(ct_images_train)
    pet_images_train_2d = flatten_3d_to_2d(pet_images_train)
    ct_images_val_2d = flatten_3d_to_2d(ct_images_val)
    pet_images_val_2d = flatten_3d_to_2d(pet_images_val)

    num_samples_train = X_train_tensor.size(0)
    # Convert to tensors
    ct_tensor_train = torch.tensor(ct_images_train_2d, dtype=torch.float32).to(device)
    ct_tensor_train = ct_tensor_train.view(num_samples_train, -1, 1, 128, 128)
    pet_tensor_train = torch.tensor(pet_images_train_2d, dtype=torch.float32).to(device)
    pet_tensor_train = pet_tensor_train.view(num_samples_train, -1, 1, 128, 128)

    num_samples_val = X_val_tensor.size(0)
    ct_tensor_val = torch.tensor(ct_images_val_2d, dtype=torch.float32).to(device)
    ct_tensor_val = ct_tensor_val.view(num_samples_val, -1, 1, 128, 128)
    pet_tensor_val = torch.tensor(pet_images_val_2d, dtype=torch.float32).to(device)
    pet_tensor_val = pet_tensor_val.view(num_samples_val, -1, 1, 128, 128)

    # Combine data into TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, ct_tensor_train, pet_tensor_train, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, ct_tensor_val, pet_tensor_val, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MultimodalModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
    criterion = CoxPHLoss()
    scaler = GradScaler()

    epochs = 300
    patience = 100
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []

    # Training loop with early stopping
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for clinical_data, ct_data, pet_data, targets in train_loader:
            clinical_data = clinical_data.to(device)
            ct_data = ct_data.to(device)
            pet_data = pet_data.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(clinical_data, ct_data, pet_data)
                # print(f"Outputs shape: {outputs.shape}")
                # print(f"Targets shape: {targets.shape}")
                # Separate survival time and event status
                survival_time = targets[:, 0]
                event_status = targets[:, 1]
                # print(f"Event status shape: {event_status.shape}")
                loss = criterion(outputs, survival_time, event_status)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for clinical_data, ct_data, pet_data, targets in val_loader:
                clinical_data = clinical_data.to(device)
                ct_data = ct_data.to(device)
                pet_data = pet_data.to(device)
                targets = targets.to(device)

                outputs = model(clinical_data, ct_data, pet_data)
                survival_time = targets[:, 0]
                event_status = targets[:, 1]
                loss = criterion(outputs, survival_time, event_status)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            save_checkpoint(epoch, model, optimizer)  # Save best model
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Final model saving
    save_checkpoint(epoch, model, optimizer, path="final_model.pth")
