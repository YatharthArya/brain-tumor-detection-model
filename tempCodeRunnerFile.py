import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Choose folder to test (either 'yes' or 'no')
input_folder = 'dataset/yes'  # change to 'dataset/no' to try non-tumor images

# Loop through all images
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # === STEP 1: Show Original ===
        plt.imshow(img_rgb)
        plt.title("Original MRI")
        plt.axis('off')
        plt.show()

        # === STEP 2: Preprocessing (Grayscale + Blur) ===
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        plt.imshow(blur, cmap='gray')
        plt.title("Blurred Image")
        plt.axis('off')
        plt.show()

        # === STEP 3: Tumor Contour Detection ===
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

        # Draw contours
        contour_img = img_rgb.copy()
        cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 2)

        plt.imshow(contour_img)
        plt.title("Detected Tumor Contours")
        plt.axis('off')
        plt.show()
