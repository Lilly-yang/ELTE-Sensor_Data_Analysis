import cv2
import os
import numpy as np


def Conv2D(img, ker, stride=1, padding=1):
    size = list(img.shape)

    pad_img = np.zeros([size[0] + 2 * padding, size[1] + 2 * padding])
    pad_img[padding:-padding, padding:-padding] = img
    img = pad_img

    out_size = [(img.shape[0] - ker.shape[0]) // stride + 1, (img.shape[1] - ker.shape[1]) // stride + 1]
    res = np.zeros(out_size)

    for hi in range(0, out_size[0] * stride, stride):
        for wi in range(0, out_size[1] * stride, stride):
            region = img[hi:hi + ker.shape[0], wi:wi + ker.shape[0]]
            res[hi // stride, wi // stride] = np.sum(region * ker[:, :])

    return res


def prewitt_filters(img):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    imgx = Conv2D(img, kernelx)
    imgy = Conv2D(img, kernely)

    magnitude = np.hypot(imgx, imgy)
    magnitude = magnitude / magnitude.max() * 255  # grayscale

    slope = np.arctan2(imgx, imgy)

    return magnitude, slope


def nms(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)

    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:  # angle 0
                q = img[i, j + 1]
                r = img[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:  # angle 45
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:  # angle 90
                q = img[i + 1, j]
                r = img[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:  # angle 135
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]

            if img[i, j] >= q and img[i, j] >= r:
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0

    return Z


if __name__ == '__main__':
    img_path = '/Users/liyang/Desktop/Course/Sensor Data Analytics (IVA)/test_data/1_prewitt'
    img_names = ['circlegrey.png', 'julia.png', 'motor.png']

    for name in img_names:
        # Load image and grayscale
        img = cv2.imread(os.path.join(img_path, name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Prewitt Edge Detection
        img_magnitude, img_slope = prewitt_filters(gray)
        cv2.imwrite(img_path + '/prewitt_' + name, img_magnitude)  # Saving the image

        # Non-Maximum Suppression
        img_nms = nms(img_magnitude, img_slope)
        cv2.imwrite(img_path + '/nms_' + name, img_nms)  # Saving the image
