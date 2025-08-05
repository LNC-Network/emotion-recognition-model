import os
import torch
from torchvision import transforms
import torch.nn as nn

def cnn_loader(model_path: str, data_dir: str = "data/train"):
    # Automatically use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get class names from training directory
    class_names = sorted(os.listdir(data_dir))
    num_classes = len(class_names)

    # Define CNN architecture
    class MoodCNN(nn.Module):
        def __init__(self, num_classes):
            super(MoodCNN, self).__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),        
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),        
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),        
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.network(x)

    # Initialize and load model
    model = MoodCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    return model, transform, class_names, device
