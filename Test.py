import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os
import csv
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
import io, base64, torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models import resnet50

# ----------------------------
# CONFIG
# ----------------------------
# Path to trained weights
WEIGHTS_PATH = r"give your model name.pth"  # Update with your model path

# Number of classes (background + dataset categories)
NUM_CLASSES = 2   # Background + GPS + Cable (adjust if needed)

# Class names (index must match your dataset's category_id)
CLASS_NAMES = ["__background__", "SignBoard"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.80

# ----------------------------
# Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = maskrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=NUM_CLASSES
)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Prediction + Visualization
# ----------------------------
def predict_and_visualize(img_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    # Run model
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    boxes, scores, labels = predictions["boxes"].tolist(), predictions["scores"].tolist(), predictions["labels"].tolist()
    # Draw predictions
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    #Custom logic to handle multiple detections and assign scores
    gps_scores, cable_scores = [], []

    for box, label, score in zip(boxes, labels, scores):
        if label == 1:
            gps_scores.append(score)
        elif label == 2:
            cable_scores.append(score)

    if len(cable_scores) == 0:
        cable_scores = [0]
    if len(gps_scores) == 0:
        gps_scores = [0]
    #lable_image = ""  
    max_gps_score = max(gps_scores)
    max_cable_score = max(cable_scores)
    for box, label, score in zip (boxes, labels, scores):
        if (score == max_gps_score and score >= CONFIDENCE_THRESHOLD) or (score == max_cable_score and score >= CONFIDENCE_THRESHOLD):
            print("scores", score)
            draw.rectangle(box, outline="red", width=3)
            print("label : ", label)
            #lable_image = label
            
            draw.text((box[0], box[1]), f"Class {CLASS_NAMES[label]}: {score:.2f}", fill="yellow", font=font)        

    # Encode image
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    if len(cable_scores) == 0:
        cable_scores = [0]
    if len(gps_scores) == 0:
        gps_scores = [0]   

    if len(cable_scores) >= 1 and max(cable_scores) >= CONFIDENCE_THRESHOLD:
        cable_score = 75
    else:
        cable_score = 0
    
    if len(gps_scores) >= 1 and max(gps_scores) >= CONFIDENCE_THRESHOLD:
        gps_score = 25
    else:
        gps_score = 0
        cable_score = 0
    total_score = gps_score + cable_score

    output_path = os.path.join(output_dir, os.path.basename(img_path))
    img.save(output_path)
    print(f"Saved result to {output_path}")

    return img_base64, max(gps_scores), max(cable_scores), gps_score, cable_score, total_score

    

# ----------------------------
# Run on test images
# ----------------------------
if __name__ == "__main__":
    
    image_test_data = [["Image Name", "GPS Cofedence Level", "Cable Cofedence Level", "GPS Score", "Cable Score", "Total Score"]]
    folder_pth = r"C:\Users\Nm_User0\Desktop\Computer Vision\new_project\Test images"
    for file_name in os.listdir(folder_pth):
        file_path = os.path.join(folder_pth, file_name)
        if os.path.isfile(file_path):
            print(file_name)
                    
            
        #for img_path in [str(file_path)]:
        img_base64, gps_scores_cf, cable_scores_cf, gps_score, cable_score, total_score = predict_and_visualize(str(file_path))

        print("gps_scores_cf", gps_scores_cf, "cable_scores_cf", cable_scores_cf)
        
        final_data_list = [file_name, gps_scores_cf, cable_scores_cf, gps_score, cable_score, total_score]
        image_test_data.append(final_data_list)

        with open("test_output.csv","w", newline = '', encoding= 'utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(image_test_data)