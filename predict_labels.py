import os
import torch
from torch.utils.data import DataLoader
from model import DGCNN_cls  # Adjust with the actual class name
from data import ModelNet40  # Adjust with the actual dataset class

class Args:
    emb_dims = 1024  # Example parameter, adjust based on your model's requirements
    dropout = 0.5    # Example parameter
    k = 20           # Example parameter

args = Args()

model = DGCNN_cls(args)

# Replace with the actual path to your model weights
'''model_weights_path = '/full/path/to/your/model_weights.pth'

if not os.path.exists(model_weights_path):
    raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

model.load_state_dict(torch.load(model_weights_path))'''
model.eval()

# In your model or script file
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


# DataLoader setup
test_dataset = ModelNet40(num_points=1024, partition='test')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Predict and collect labels
all_predictions = []
for inputs, _ in test_loader:
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.tolist())

print("Classified Labels:", all_predictions)
