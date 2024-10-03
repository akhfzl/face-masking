import numpy as np
import cv2
import torch
import sys
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

yolov5_path = pathlib.PosixPath('yolov5')
sys.path.append(str(yolov5_path))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes

model = attempt_load('best.pt')
model.eval()
pathlib.PosixPath = temp

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to 640x640
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]  # Convert BGR to RGB
    img = np.ascontiguousarray(img)  # Make sure the array is contiguous
    img = torch.from_numpy(img).float()  # Convert to Tensor
    img /= 255.0  # Normalize to [0, 1]

    # Change the shape to [batch_size, channels, height, width]
    img = img.permute(2, 0, 1).unsqueeze(0)  # Rearrange dimensions and add batch dimension

    # Perform inference
    with torch.no_grad():
        pred = model(img)[0]  # Get predictions

    # Apply Non-Maximum Suppression
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Process predictions
    for det in pred:  # detections per image
        if det is not None and len(det):
            # Rescale boxes from 640 to original frame size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # Draw bounding boxes on the frame
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]}: {conf:.2f}'  # Create label
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)  # Draw box
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Draw label

    # Display the frame
    cv2.imshow('Deteksi Cadar Masker', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()