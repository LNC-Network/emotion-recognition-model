import os
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

def cnn_loader(model_path: str, data_dir: str = "data/train"):
    # Automatically use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get class names from training directory
    class_names = sorted(os.listdir(data_dir))
    num_classes = len(class_names)

    # Define CNN architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
        
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input_shape=(1,48,48)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(0.25)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # padding=2 for 'same' with kernel=5
            self.bn2 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.dropout2 = nn.Dropout(0.25)

            self.conv4 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(512)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.dropout3 = nn.Dropout(0.25)

            self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.pool4 = nn.MaxPool2d(2, 2)
            self.dropout4 = nn.Dropout(0.25)

            self.flatten_dim = 512 * 3 * 3  # After 4 maxpools on 48x48 input: 48->24->12->6->3

            self.fc1 = nn.Linear(self.flatten_dim, 256)
            self.bn5 = nn.BatchNorm1d(256)
            self.dropout5 = nn.Dropout(0.25)

            self.fc2 = nn.Linear(256, 512)
            self.bn6 = nn.BatchNorm1d(512)
            self.dropout6 = nn.Dropout(0.25)

            self.fc3 = nn.Linear(512, 7)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.bn1(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            
            x = F.relu(self.conv3(x))
            x = self.bn2(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            
            x = F.relu(self.conv4(x))
            x = self.bn3(x)
            x = self.pool3(x)
            x = self.dropout3(x)
            
            x = F.relu(self.conv5(x))
            x = self.bn4(x)
            x = self.pool4(x)
            x = self.dropout4(x)
            
            x = x.view(-1, self.flatten_dim)  # Flatten
            
            x = F.relu(self.fc1(x))
            x = self.bn5(x)
            x = self.dropout5(x)
            
            x = F.relu(self.fc2(x))
            x = self.bn6(x)
            x = self.dropout6(x)
            
            x = self.fc3(x)
            return F.log_softmax(x, dim=1) 

    # Initialize and load model
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    return model, transform, class_names, device
