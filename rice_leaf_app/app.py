from flask import Flask, request, render_template
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import base64
import re
from io import BytesIO
import numpy as np

app = Flask(__name__)

# === Load model (ProtoNet)
class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder

    def forward(self, support, support_labels, query, n_way):
        support_embed = self.encoder(support)
        query_embed = self.encoder(query)

        prototypes = []
        for c in range(n_way):
            prototypes.append(support_embed[support_labels == c].mean(0))
        prototypes = torch.stack(prototypes)

        dists = torch.cdist(query_embed, prototypes)
        probs = (-dists).softmax(dim=1)
        return probs, dists

# === Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Load model
resnet = resnet18(weights=None)
resnet.fc = nn.Identity()
model = ProtoNet(resnet).to(device)
model.load_state_dict(torch.load("model/final_fewshot_model.pth", map_location=device))
model.eval()

# Load support set
support_x, support_y = torch.load("model/support_set_fixed.pt")
support_x, support_y = support_x.to(device), support_y.to(device)

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Improved thresholds for better classification
NON_RICE_LEAF_THRESHOLD = 18.0  # Adjusted threshold for non-rice leaf detection
HEALTHY_LEAF_THRESHOLD = 12.0   # Threshold for healthy leaf detection
MIN_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for classification

# Function to analyze green content in image (rice leaves are typically green)
def analyze_green_content(img):
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Extract RGB channels
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Calculate green ratio (green channel dominance)
    # Higher values indicate more green content
    green_ratio = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-6)
    
    # Calculate green dominance (how much green exceeds other channels)
    green_dominance = np.mean((g > r) & (g > b))
    
    return green_ratio, green_dominance

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img = None
        
        # Handle file upload
        if 'image' in request.files and request.files['image'].filename:
            img_file = request.files["image"]
            img = Image.open(img_file).convert("RGB")
        
        # Handle camera data
        elif 'camera_data' in request.form and request.form['camera_data']:
            camera_data = request.form['camera_data']
            # Remove the data URL prefix
            image_data = re.sub('^data:image/jpeg;base64,', '', camera_data)
            img = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
        
        if img:
            # Analyze color characteristics (rice leaves are typically green)
            green_ratio, green_dominance = analyze_green_content(img)
            
            # Convert for model input
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs, dists = model(support_x, support_y, img_tensor, n_way=3)
                
                # Get minimum distance to any prototype
                min_dist = dists.min().item()
                
                # Get the predicted class and its confidence
                pred = probs.argmax(1).item()
                confidence_value = probs[0][pred].item()
                
                # Get all distances for better analysis
                all_distances = [d.item() for d in dists[0]]
                
                print(f"Debug - All distances: {all_distances}")
                print(f"Debug - Green ratio: {green_ratio}, Green dominance: {green_dominance}")
                
                # Improved classification logic combining multiple factors
                is_rice_leaf = True
                
                # Check if it's NOT a rice leaf (high distance AND low confidence AND low green content)
                if ((min_dist > NON_RICE_LEAF_THRESHOLD) or 
                    (confidence_value < MIN_CONFIDENCE_THRESHOLD) or
                    (green_ratio < 1.0 and green_dominance < 0.5)):
                    
                    # Additional check: if all distances are high, it's likely not a rice leaf
                    if min(all_distances) > NON_RICE_LEAF_THRESHOLD * 0.8:
                        result = "Not a Rice Leaf"
                        confidence = "100%"
                        is_rice_leaf = False
                
                # If it is a rice leaf, determine if it's healthy or diseased
                if is_rice_leaf:
                    # Check if it's a healthy rice leaf
                    if min_dist > HEALTHY_LEAF_THRESHOLD and green_dominance > 0.7:
                        result = "Healthy Rice Leaf"
                        confidence = "95%"
                    else:
                        # It's a diseased rice leaf
                        result = class_names[pred]
                        confidence = f"{confidence_value:.2%}"
                
                # Debug information to help with threshold tuning
                print(f"Debug - Min Distance: {min_dist}, Confidence: {confidence_value}, Result: {result}")

            return render_template("index.html",
                                result=result,
                                confidence=confidence,
                                is_rice_leaf=is_rice_leaf)
        
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)