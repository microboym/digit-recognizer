import cv2
from matplotlib import pyplot as plt
import numpy as np

def accessBinary(img, roi, threshold = 165, kernel_size = (2, 2)):
    # Crop
    x, y, w, h = roi
    if roi != (0, 0, 0, 0):
        img = img[y:y+h, x:x+w]

    # Grey
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = grey.shape[0]
    w = grey.shape[1]

    # Thresh
    _, thresh = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY_INV)

    # Dilate
    kernel = np.ones(kernel_size, np.uint8)
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    return (img, thresh, dilate)

def get_borders(bin, length, min_size=150):
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = lambda size: size[0] * size[1]
    contours.sort(key=lambda c: area(cv2.boundingRect(c)[2:]), reverse=True)
    contours = contours[:length]

    borders = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > min_size:
            border = [(x, y), (x+w, y+h)]
            borders.append(border)
    borders.sort(key=lambda x: x[0][0], reverse=False)
    return borders

def extract_numbers(bin, borders, size=(28, 28)):
    img_data = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    for i, border in enumerate(borders):
        border_img = bin[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        extend_piexl = (max(border_img.shape) - min(border_img.shape)) // 2
        target_img = cv2.copyMakeBorder(border_img, 7, 7, extend_piexl + 7, extend_piexl + 7, cv2.BORDER_CONSTANT)
        target_img = cv2.resize(target_img, size)
        target_img = np.expand_dims(target_img, axis=-1)
        img_data[i] = target_img
    return img_data

def draw_border(img, borders):
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
    return img

if __name__ == '__main__':
    length = 5

    # Open image
    path = input("Input path: ")
    image = cv2.imread(path)

    if image is not None:
        roi = cv2.selectROI("Crop", image)
        print("Got ROI: ", roi)
    else:
        # Open default test image
        path = "./test_data/1.jpg"
        image = cv2.imread(path)
        roi = (400, 716, 124, 117)
    
    # Access binary
    crop, thresh, binary = accessBinary(image, roi = roi)
    plt.subplot(2, 2, 1), plt.imshow(binary, "gray"), plt.title("Binary")

    # Get borders
    borders = get_borders(binary, length=length)
    bin_border = crop.copy()

    print("Got borders:", borders)
    bin_border = draw_border(bin_border, borders)
    plt.subplot(2, 2, 2), plt.imshow(bin_border, "gray"), plt.title("Borders")

    # Extract numbers from the borders
    data = extract_numbers(binary, borders)
    if len(data) == length:
        for i in range(length):
            plt.subplot(2, length, length+i+1)
            plt.imshow(data[i]), plt.title("Number {}".format(i+1))

    plt.show()