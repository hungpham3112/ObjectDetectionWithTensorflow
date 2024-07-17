import argparse
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import time
import pandas as pd

def load_dataset(dataset_path):
    try:
        dataset = pd.read_csv(dataset_path)
        if 'label' not in dataset.columns or 'name' not in dataset.columns or 'price' not in dataset.columns:
            raise ValueError("Dataset file must contain 'label', 'name', and 'price' columns.")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None, None
    
    class_names = dataset['label'].astype(str).tolist()
    label_to_name = dict(zip(dataset['label'], dataset['name']))
    label_to_price = dict(zip(dataset['label'], dataset['price']))
    
    return dataset, class_names, label_to_name, label_to_price

def main(model_path, dataset_path):
    np.set_printoptions(suppress=True)

    # Load dataset
    dataset, class_names, label_to_name, label_to_price = load_dataset(dataset_path)
    if dataset is None:
        return

    model = load_model(model_path, compile=False)

    cap = cv2.VideoCapture(0)

    current_class = None
    start_time = None
    recognized_classes = []
    high_confidence_recognized = False
    cart = {}  # Dictionary to store recognized items and their counts

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        image = Image.fromarray(frame[..., ::-1])
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        label = int(index)  # Ensure label is integer
        if label in label_to_name:
            class_name = label_to_name[label]
        else:
            class_name = f"Unknown-{label}"

        confidence_score = prediction[0][index]

        cv2.putText(
            frame,
            "Class: " + class_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Confidence Score: " + str(confidence_score),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        current_time = time.time()

        if confidence_score > 0.9999:
            if current_class == class_name:
                if not high_confidence_recognized:
                    # Check if the same class has been recognized for at least 2 seconds
                    if start_time and current_time - start_time >= 2:
                        if label in cart:
                            cart[label]["count"] += 1
                        else:
                            cart[label] = {"name": class_name, "count": 1}
                        recognized_classes.append(label)
                        high_confidence_recognized = True
                        print(f"Class '{class_name}' recognized and added to the cart.")
            else:
                # New class detected or high confidence after low confidence
                current_class = class_name
                start_time = current_time
                high_confidence_recognized = False
        else:
            # Reset tracking state when confidence drops below threshold
            current_class = None
            start_time = None
            high_confidence_recognized = False

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate total prices for recognized classes
    total_cost_all_products = 0.0
    print("\nRecognized products and their details:")
    print("===================================")
    print("Name of Product    | # of Product | Total Price")
    print("-----------------------------------")
    for label, info in cart.items():
        name = info["name"]
        count = info["count"]
        price = label_to_price.get(label, 0.0)
        total_cost = price * count
        total_cost_all_products += total_cost
        print(f"{name:<18} | {count:^12} | ${total_cost:.2f}")

    # Display total cost of all products
    print("-----------------------------------")
    print(f"Total cost of all recognized products: ${total_cost_all_products:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time object detection with a pre-trained model."
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the pre-trained model file"
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Path to the dataset CSV file"
    )

    args = parser.parse_args()
    main(args.model_path, args.dataset_path)
