from utility.loader import cnn_loader

model, transformer, class_names, device = cnn_loader("mood_cnn.pth","data/test")

from utility.camera import cv2ModelRunner

cv2ModelRunner(model, class_names, transformer, device)
