from PIL import Image, ImageDraw
from math import pi, cos, sin
from canny import canny_edge_detector
from collections import defaultdict
import os
import numpy as np
import cv2


def accumulate_image(input_image, acc):
    """
    This function is used to generate an accumulation image
    """
    acc_img = np.zeros(input_image.size)
    for k, v in acc.items():
        x, y, r = k  # (x,y) is coordinate, v is frequency
        try:
            acc_img[x, y] = v
        except:  # ignore it if (x,y) not in range of original image size
            pass

    acc_img = acc_img / acc_img.max() * 255  # scale data to 0-255

    return acc_img


def Circle_HT(input_image, d_range, smoothing):
    """
    Circle Hough Transform
    :param d_range: range of diameters
    :param smoothing: remove noise
    :return: coordinate of circles, accumulation matrix
    """
    # Generate probe circle
    points = []
    for r in range(int(d_range[0] / 2), int(d_range[1] / 2 + 1)):  # use value of radius
        for t in range(steps):
            theta = 2 * pi * t / steps
            points.append((r, int(r * cos(theta)), int(r * sin(theta))))

    # accumulation matrix
    acc = defaultdict(int)
    for x, y in canny_edge_detector(input_image, smoothing=smoothing):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    # Filter circle by threshold
    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k  # (x,y) is coordinate, v is frequency
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in
                                          circles):  # Select the circles with higher frequency and not included by other circles
            circles.append((x, y, r))

    return circles, acc


# parameters
img_path = '/Users/liyang/Desktop/Course/SensorDataAnalytics(IVA)/test_data/3_hough'
img_para = {'blood.png': [[18, 20], True], 'cable.png': [[44, 50], False], 'cells.png': [[20, 38], False],
            'circles.png': [[26, 26], False]}  # {name: [d-range, smoothing]}
steps = 60  # Number of sampling points of contour for calculation
threshold = 0.4  # coincidence rate of sampling points (to determine whether it is a circle)

if __name__ == "__main__":
    for name, para in img_para.items():
        print('---Start Circle Hough Transform---\n', os.path.join(img_path, name))
        d_range, smoothing = para

        # Load image
        input_image = Image.open(os.path.join(img_path, name))

        output_image = Image.new("RGB", input_image.size)
        output_image.paste(input_image)
        draw_result = ImageDraw.Draw(output_image)

        # Circle Hough Transform
        circles, acc = Circle_HT(input_image, d_range, smoothing)

        # save accumulation matrix to image
        acc_img = accumulate_image(input_image, acc)
        cv2.imwrite(img_path + '/acc_' + str(d_range[0]) + '_' + str(d_range[1]) + '_' + name, acc_img)

        # Draw detected circles to image and save it
        for x, y, r in circles:
            draw_result.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))

        print('Done! Save to:\n %s \n' % (img_path + '/obj_det_' + name))
        output_image.save(img_path + '/obj_det_' + str(d_range[0]) + '_' + str(d_range[1]) + '_' + name)
