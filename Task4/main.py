from opticFlow import *
from trackerTools import *
import glob


def readVedio(path):
    # Read video
    video = cv2.VideoCapture(path)

    # Exit if video not opened
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    return video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument parser for tracker algrithm')
    parser.add_argument('--inputPath', type=str, help='path to vedio file', default=None)
    parser.add_argument('--inputFile', type=str, help='name of tracker', default=None)
    parser.add_argument('--outputPath', type=str, help='path to vedio file', default=None)
    parser.add_argument('--tracker', type=str, help='name of tracker', default=None)
    parser.add_argument('--opticFlow', type=str, help='name of optic flow', default=None)
    parser.add_argument('--maxDisplacement', type=float, help='maximum length of optic flow arrow line',
                        default=20)
    parser.add_argument('--save', type=str, help='name of optic flow', default=False)
    parser.add_argument('--imshow', type=str, help='name of optic flow', default=False)
    args = parser.parse_args()

    if args.inputPath != None:
        file_list = glob.glob(args.inputPath + '/*.*')
    elif args.inputFile != None:
        file_list = [args.inputFile]
    else:
        print('Please give a path of file for director')
        exit()

    for ind, file_path in enumerate(file_list):
        print('---processing %d / %d : %s' % (ind+1, len(file_list), file_path))
        args.inputFile = file_path
        cap = readVedio(file_path)
        if args.opticFlow != None:
            if args.opticFlow == 'lk':
                LKOpticFlow(cap, args)
            elif args.opticFlow == 'dense':
                denseOpticFlow(cap, args)

        if args.tracker != None:
            if args.tracker == 'csrt':
                csrtTracker(cap)
            if args.tracker == 'medianflow':
                medianFlowTracker(cap)
            if args.tracker == 'tld':
                TLDTracker(cap)
