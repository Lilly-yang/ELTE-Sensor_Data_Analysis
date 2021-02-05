import cv2 as cv
import argparse
from flow_display import *
import os


def denseOpticFlow(cam, args):
    ret, prev = cam.read()
    if not ret:
        print('No frame in the file')
        exit()

    if args.save:
        file_name = os.path.basename(args.inputFile).split('.')[0]
        if args.outputPath != None:
            output_file = os.path.join(args.outputPath, file_name + '.avi')
        else:
            output_file = os.path.join(os.path.split(args.inputFile)[0],
                                       args.inputFile.split('/', '.')[-2] + '_opt_flow.avi')

        h, w, _ = prev.shape
        out = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'DIVX'), 10, (w, h), True)

    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    while (1):
        ret, img = cam.read()
        if not ret:
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2,
                                           0)  # finds an optical flow for each prev pixel
        prevgray = gray

        # draw the arrow line
        h, w = gray.shape[:2]
        step = args.maxDisplacement
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(np.around(lines))

        for l in lines:
            if l[0][0] - l[1][0] > 1 or l[0][1] - l[1][1] > 1:
                cv.arrowedLine(img, tuple(l[0]), tuple(l[1]), (0, 255, 255), tipLength=0.5)

        if args.save:
            out.write(img)
        if args.imshow:
            cv.imshow('flow', img)

    if args.save:
        print('save to %s' % output_file)

    out.release()
    cv.destroyAllWindows()


def LKOpticFlow(cap, args):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print('No frame in the file')
        exit()

    if args.save:
        file_name = os.path.basename(args.inputFile).split('.')[0]
        if args.outputPath != None:
            output_file = os.path.join(args.outputPath, file_name + '.avi')
        else:
            output_file = os.path.join(os.path.split(args.inputFile)[0],
                                       args.inputFile.split('/', '.')[-2] + '_opt_flow.avi')

        h, w, _ = old_frame.shape
        out = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'DIVX'), 10, (w, h), True)

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    while (1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None:
            continue
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            frame = cv.circle(frame, (a, b), 10, color[i].tolist(), -1)

        if args.save:
            out.write(frame)
        if args.imshow:
            cv.imshow('frame', frame)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    if args.save:
        print('save to %s' % output_file)

    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This sample demonstrates Optical Flow calculation')
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()

    cap = cv.VideoCapture(args.image)

    LKOpticFlow(cap)
    # denseOpticFlow(cap)
