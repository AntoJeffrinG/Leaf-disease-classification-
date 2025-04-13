from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import os

app = Flask(__name__)

# Load Model
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('leaf_disease_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return 'Diseased' if predicted.item() == 0 else 'Healthy'

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)
            prediction = predict_image(image)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
