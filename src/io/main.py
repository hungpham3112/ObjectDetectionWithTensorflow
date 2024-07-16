import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8s.pt")

# Load your image
image_path = r"C:\Users\sofia\Downloads\watch-samples\467.jpg"
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Extract the results
result = results[0]  # Access the first result in the list

# Extract bounding boxes, scores, and class IDs
boxes = result.xyxy.cpu().numpy()  # Bounding boxes
scores = result.scores.cpu().numpy()  # Confidence scores
classes = result.classes.cpu().numpy()  # Class IDs

# Convert BGR image to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot using matplotlib
plt.imshow(image_rgb)
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box[:4]
    label = f"{model.names[int(classes[i])]} {scores[i]:.2f}"
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2)
    )
    plt.text(
        x1,
        y1,
        label,
        color="red",
        fontsize=12,
        bbox=dict(facecolor="yellow", alpha=0.5),
    )
plt.axis("off")
plt.show()
