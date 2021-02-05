import cv2
import os
import numpy as np


def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0 / pixel_number

    # get histogram, the parameter bins is [0,1), [1,2),...,[254,255),[255,256]
    his, bins = np.histogram(gray, np.arange(0, 257))  # len(his) = 256, bins = [0,1,...,256]

    final_thresh = -1
    final_value = -1

    intensity_arr = np.arange(256)  # [0,1,...,255]

    for t in bins[1:-2]:  # This goes from 1 to 255
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        np.seterr(divide='ignore', invalid='ignore')  # ignore warning when dived by 0 or Nan

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)

        value = Wb * Wf * (mub - muf) ** 2

        # get the optimal threshold value t
        if value > final_value:
            final_thresh = t
            final_value = value

    print(final_thresh)

    final_img = gray.copy()
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0

    return final_img, final_thresh


if __name__ == '__main__':
    img_path = '/Users/liyang/Desktop/Course/Sensor Data Analytics (IVA)/test_data/2_otsi'
    img_names = ['aluminium.png', 'finger.png', 'julia.png', 'phobos.png']

    for name in img_names:
        # Load image and grayscale
        img = cv2.imread(os.path.join(img_path, name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu threshold
        img_otsu, t = otsu(gray)
        cv2.imwrite(img_path + '/otsu_' + str(t) + '_' + name, img_otsu)  # Saving the image
