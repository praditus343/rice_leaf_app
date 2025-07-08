from flask import Flask, request, render_template
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import base64
import re
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Logging aktif
import logging
logging.basicConfig(level=logging.INFO)

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
model.load_state_dict(torch.load("final_fewshot_model.pth", map_location=device))
model.eval()

# Load support set
support_x, support_y = torch.load("support_set_fixed.pt", map_location=device)
support_x, support_y = support_x.to(device), support_y.to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),  # Ambil 3 channel RGB
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

NON_RICE_LEAF_THRESHOLD = 18.0
HEALTHY_LEAF_THRESHOLD = 12.0
MIN_CONFIDENCE_THRESHOLD = 0.4

def analyze_green_content(img):
    img_array = np.array(img)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    green_ratio = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-6)
    green_dominance = np.mean((g > r) & (g > b))
    return green_ratio, green_dominance

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    is_rice_leaf = None

    if request.method == "POST":
        try:
            img = None

            if 'image' in request.files and request.files['image'].filename:
                img_file = request.files["image"]
                img = Image.open(img_file).convert("RGB")
            elif 'camera_data' in request.form and request.form['camera_data']:
                camera_data = request.form['camera_data']
                image_data = re.sub('^data:image/jpeg;base64,', '', camera_data)
                img = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")

            if img:
                green_ratio, green_dominance = analyze_green_content(img)
                img_tensor = transform(img).unsqueeze(0).to(device)

                logging.info(f"[INFO] Input image shape: {img_tensor.shape}")
                logging.info(f"[INFO] Green Ratio: {green_ratio:.2f}, Green Dominance: {green_dominance:.2f}")

                with torch.no_grad():
                    probs, dists = model(support_x, support_y, img_tensor, n_way=3)
                    min_dist = dists.min().item()
                    pred = probs.argmax(1).item()
                    confidence_value = probs[0][pred].item()
                    all_distances = [d.item() for d in dists[0]]

                    is_rice_leaf = True
                    if ((min_dist > NON_RICE_LEAF_THRESHOLD) or 
                        (confidence_value < MIN_CONFIDENCE_THRESHOLD) or
                        (green_ratio < 1.0 and green_dominance < 0.5)):
                        if min(all_distances) > NON_RICE_LEAF_THRESHOLD * 0.8:
                            result = "Not a Rice Leaf"
                            confidence = "100%"
                            is_rice_leaf = False

                    if is_rice_leaf:
                        if min_dist > HEALTHY_LEAF_THRESHOLD and green_dominance > 0.7:
                            result = "Healthy Rice Leaf"
                            confidence = "95%"
                        else:
                            result = class_names[pred]
                            confidence = f"{confidence_value:.2%}"

        except Exception as e:
            logging.error(f"[ERROR] Saat memproses gambar: {e}")
            result = "Gagal menganalisis gambar"
            confidence = "0%"
            is_rice_leaf = False

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           is_rice_leaf=is_rice_leaf)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
