from utility.loader import cnn_loader
from utility.camera import cv2ModelRunner


model, transformer, class_names, device = cnn_loader("models/model.pth","/home/jit/.cache/kagglehub/datasets/msambare/fer2013/versions/1/train")

cv2ModelRunner(model, class_names, transformer, device)
