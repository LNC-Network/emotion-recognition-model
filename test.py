import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import torch.nn as nn

# ========== Config ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['happy', 'sad', 'neutral', 'angry', 'surprised', 'disgust', 'fear']  # 7 classes

# ========== Model ==========
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ========== Load Model ==========
model = ImageClassifier().to(device)
model.load_state_dict(torch.load("image_classifier.pth", map_location=device))
model.eval()

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])

# ========== Start Webcam ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("🟢 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        mood = class_names[predicted.item()]

    # Show prediction in terminal
    print(f"Prediction: {mood}", end="\r", flush=True)

    # Display window
    cv2.putText(frame, f"Mood: {mood}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow('Live Mood Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
