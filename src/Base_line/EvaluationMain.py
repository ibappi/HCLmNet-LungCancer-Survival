import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss, confusion_matrix
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from TrainModel import MultimodalModel, device  # Import your updated model class

# Load the saved model
model_path = './data/final_model.pth'
checkpoint = torch.load(model_path, map_location=device)

# Initialize the model with updated architecture
model = MultimodalModel()  # Ensure this matches your latest model class
model_state_dict = model.state_dict()

# Load only the compatible keys, skipping missing or mismatched ones
filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_state_dict}
model.load_state_dict(filtered_state_dict, strict=False)  # Set strict=False to ignore missing keys

model.to(device)  # Move the model to the appropriate device
model.eval()

# Load the evaluation data
test_clinical_data = pd.read_excel('./data/output/X_clinical_test.xlsx')
true_labels = pd.read_excel('./data/output/y_clinical_test.xlsx')  # Load the true labels

# Extract ground truth survival times and event data
true_survival_times = true_labels['Survival.time'].values
true_event_observed = true_labels['Deadstatus.event'].values

CT_data = np.load('./data/output/ct_images_test.npy')
PET_data = np.load('./data/output/pet_images_test.npy')


# Process data as per your model input
def preprocess_data(clinical_data, CT_data, PET_data):
    # Select numeric clinical data columns
    numeric_clinical_data = clinical_data.select_dtypes(include=[np.number])

    # Convert clinical data to tensor
    clinical_data_tensor = torch.tensor(numeric_clinical_data.values, dtype=torch.float32).to(device)

    # Convert CT and PET data to tensors and ensure proper dimensions
    CT_data_tensor = torch.tensor(CT_data, dtype=torch.float32).to(device)
    PET_data_tensor = torch.tensor(PET_data, dtype=torch.float32).to(device)

    if CT_data_tensor.dim() == 4:  # Add a channel dimension if necessary
        CT_data_tensor = CT_data_tensor.unsqueeze(1)
    if PET_data_tensor.dim() == 4:  # Add a channel dimension if necessary
        PET_data_tensor = PET_data_tensor.unsqueeze(1)

    return clinical_data_tensor, CT_data_tensor, PET_data_tensor


clinical_tensor, CT_tensor, PET_tensor = preprocess_data(test_clinical_data, CT_data, PET_data)


# Evaluation function
def calculate_c_index(true_survival_times, predicted_risk, event_observed):
    return concordance_index(true_survival_times, predicted_risk, event_observed)


# Update evaluation function to work with the new model architecture
def evaluate_model_with_5_year_prediction(model, clinical_tensor, CT_tensor, PET_tensor, true_survival_times, true_event_observed, prediction_time=5):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predicted_risk = model(clinical_tensor, CT_tensor, PET_tensor)

    predicted_risk = predicted_risk.cpu().numpy()  # Convert tensor to NumPy array if needed

    # Ensure predicted_risk is in a reasonable range
    print("Predicted risk values:", predicted_risk)  # Debug: Check values

    c_index = calculate_c_index(true_survival_times, predicted_risk, true_event_observed)
    mae = mean_absolute_error(true_survival_times, predicted_risk)

    # Apply min-max normalization to predicted risk values
    predicted_risk_normalized = (predicted_risk - predicted_risk.min()) / (predicted_risk.max() - predicted_risk.min())

    # Clip to avoid exceeding 1 for probabilities
    predicted_risk_normalized = np.clip(predicted_risk_normalized, 0, 1)

    # Calculate Brier Score using the normalized predicted risk
    brier_score = brier_score_loss(true_event_observed, predicted_risk_normalized)

    # Threshold and classification
    threshold = 0.5
    predicted_classes = (predicted_risk_normalized > threshold).astype(int)

    accuracy = accuracy_score(true_event_observed, predicted_classes)
    conf_matrix = confusion_matrix(true_event_observed, predicted_classes)

    # Assuming predicted risk is log-hazard, convert to 5-year survival probability
    survival_probabilities_5_year = np.exp(-predicted_risk_normalized * prediction_time)  # Time in years
    survival_probabilities_5_year = np.clip(survival_probabilities_5_year, 0, 1)  # Ensure probabilities are in [0, 1]

    return c_index, mae, brier_score, accuracy, conf_matrix, predicted_risk, survival_probabilities_5_year

# Perform evaluation with the updated model
c_index, mae, brier_score, accuracy, conf_matrix, predicted_risk, survival_probabilities_5_year = evaluate_model_with_5_year_prediction(
    model, clinical_tensor, CT_tensor, PET_tensor, true_survival_times, true_event_observed
)

# Print the results
print(f"C-Index: {c_index:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Brier Score: {brier_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap='Blues', fignum=1)
plt.title('Confusion Matrix', pad=20)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(ticks=[0, 1], labels=['Event Not Observed', 'Event Observed'])
plt.yticks(ticks=[0, 1], labels=['Event Not Observed', 'Event Observed'])
plt.grid()
plt.show()

# Plot Survival Probability
plt.figure(figsize=(10, 6))
true_survival_times_years = true_survival_times / 365.0  # Convert days to years

for i in range(len(survival_probabilities_5_year)):
    plt.plot([0, true_survival_times_years[i]], [1, survival_probabilities_5_year[i]], marker='o', label=f'Patient {i+1}', alpha=0.7)

plt.xlabel('Survival Time (Years)')
plt.ylabel('Survival Probability')
plt.title('Individual Patient Survival Probability Over Time')
plt.ylim(0, 1)  # Fix y-axis limits to [0, 1]
plt.grid()
plt.legend()
plt.show()
