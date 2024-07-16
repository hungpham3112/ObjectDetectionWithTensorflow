import argparse
from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np
import cv2

def main(model_path):
    np.set_printoptions(suppress=True)

    model = load_model(model_path, compile=False)

    class_names = open("labels.txt", "r").readlines()

    cap = cv2.VideoCapture(0)  

    while(True):
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        image = Image.fromarray(frame[...,::-1])  
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        cv2.putText(frame, "Class: " + class_name[2:-1], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Confidence Score: " + str(confidence_score), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time object detection with a pre-trained model.')
    parser.add_argument('model_path', type=str, help='Path to the pre-trained model file')

    args = parser.parse_args()
    main(args.model_path)
