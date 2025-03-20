import cv2
import torch
import numpy as np
import pandas as pd

# --- Data Engineering and Deep Learning Setup ---
# 1. Load the pretrained YOLOv5 model from PyTorch Hub.
#    (Note: For production, you might fine-tune a custom model on your product dataset.)
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # set confidence threshold

# 2. Simulated product details database.
#    For a production system, this might come from a database or a dedicated service.
#    Here, we map some common COCO labels to product details.
product_details_db = {
    "bottle": {"brand": "Coca-Cola", "product_name": "Coke 500ml"},
    "cup": {"brand": "Generic", "product_name": "Paper Cup"},
    "book": {"brand": "Penguin", "product_name": "Novel"},
    # Add more mappings as needed...
}

# --- Prototype Functionality ---
def capture_image():
    """Capture a single frame from the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Failed to capture image.")
        return None
    return frame

def annotate_frame(frame, detections):
    """Annotate the frame with detection boxes and labels."""
    for det in detections:
        # det: [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, text, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def count_products(detections):
    """
    Count detected products and lookup product details.
    Returns a dictionary: {label: {"count": n, "details": {…} or None}}.
    """
    counts = {}
    for det in detections:
        # det: [x1, y1, x2, y2, confidence, class]
        cls_idx = int(det[5])
        label = model.names[cls_idx]
        if label in counts:
            counts[label]["count"] += 1
        else:
            # Lookup product details; if not found, set details to None
            details = product_details_db.get(label, None)
            counts[label] = {"count": 1, "details": details}
    return counts

def main():
    # Capture an image from the webcam
    frame = capture_image()
    if frame is None:
        return

    # Convert image to RGB (YOLOv5 expects images in RGB format)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference using YOLOv5 model
    results = model(img_rgb)
    # results.xyxy[0] returns detections in format [x1, y1, x2, y2, conf, cls]
    detections = results.xyxy[0].cpu().numpy()

    # Count products and lookup details
    product_counts = count_products(detections)
    print("Detected items and counts:")
    for label, info in product_counts.items():
        details = info["details"]
        if details:
            print(f"Label: {label} | Count: {info['count']} | Brand: {details['brand']}, Product: {details['product_name']}")
        else:
            print(f"Label: {label} | Count: {info['count']} | No detailed product info available.")

    # Annotate frame with detection boxes and labels
    annotated_frame = annotate_frame(frame.copy(), detections)

    # Display the annotated image (press any key to close)
    cv2.imshow("Detected Products", annotated_frame)
    print("Press any key on the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
