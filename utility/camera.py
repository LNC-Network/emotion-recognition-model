import cv2
from PIL import Image
import torch
from torchvision import transforms


def cv2ModelRunner(model, class_names, transform, device):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam.")
        exit()

    print("🟢 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
          
        # Convert OpenCV image to PIL and preprocess
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)
    
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            mood = class_names[predicted.item()]
    
        # Display mood on frame
        cv2.putText(frame, f"Mood: {mood}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Mood Detection', frame)
    
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    

