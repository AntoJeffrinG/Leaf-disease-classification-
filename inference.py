from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

# Step 1: Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Step 2: Recreate model architecture
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classifier: healthy vs diseased

# Step 3: Load trained weights
model.load_state_dict(torch.load('leaf_disease_classifier.pth', map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Step 4: Define inference function
def predict_leaf_disease(image_path):
    """
    Predicts whether a leaf is healthy or diseased.

    Args:
        image_path (str): Path to the input leaf image.

    Returns:
        str: 'Healthy' or 'Diseased'
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return 'Diseased' if predicted.item() == 0 else 'Healthy'

# Step 5: Run inference
image_path = '/Users/jeffrin/Desktop/Insect/dataset/val/diseased/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG'
result = predict_leaf_disease(image_path)
print(f"Prediction: {result}")
