import torch
from torch.utils.data import DataLoader
# Make sure you import your model and any necessary configurations here

model.eval()  # Set the model to evaluation mode

test_loader = DataLoader(...)  # Ensure you have correctly set up your DataLoader

all_predictions = []
for inputs, _ in test_loader:
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.tolist())

print("Classified Labels:", all_predictions)
