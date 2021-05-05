import cv2
import keras
import numpy as np
import pre_process
import sys, os

model = keras.models.load_model("model.h5")

def get_numbers(img, roi, length=5):
    _, _, bin = pre_process.accessBinary(img, roi)
    borders = pre_process.get_borders(bin, length=length)
    number_images = pre_process.extract_numbers(bin, borders)

    results = model.predict(number_images)
    numbers = []
    for res in results:
        numbers.append(np.argmax(res))
    return borders, numbers

if __name__ == "__main__":
    # Open image
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        images = [cv2.imread(path) for path in paths]
    else:
        paths = os.listdir("test_data/")
        images = [cv2.imread("test_data/" + path) for path in paths]

    roi = (1583, 1706, 319, 124)
    length = 5
    for index, image in enumerate(images):
        if image is not None:
            roi = cv2.selectROI("Crop", image)
            borders, numbers = get_numbers(image, roi, length=length)
            print(paths[index], "Numbers: ", numbers)
                