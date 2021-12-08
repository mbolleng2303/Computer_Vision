# Import required modules
import cv2
import numpy as np
import os
import glob

#global variable
pts = []


def get_path(name, output=False):
    parent_dir = os.getcwd()
    if not output:
        for file in glob.glob(os.path.join(parent_dir, "Input\\*")):
            if os.path.basename(os.path.normpath(file)) == name:
                return file
        else:
            print("This files doesn't exist")
    else:
        path = os.path.join(parent_dir, 'Output')
        file_dir = os.path.join(path, name)
        if os.path.isfile(file_dir):
            os.remove(file_dir)
            return file_dir
        else:
            return file_dir


def get_points(text_file):
    file_path = get_path(text_file)
    with open(file_path, "r") as file:
        lines = file.readlines()
        points = np.loadtxt(lines)
    return points


def click_event(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))


def store_click(image, file_name):
    global pts
    # reading the image
    img = cv2.imread(image, 1)
    # displaying the image
    cv2.imshow('image', img)
    # setting mouse handler for the image
    # and calling the click_event() function

    cv2.setMouseCallback('image', click_event)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # close the window

    pts = np.array(pts)
    np.savetxt(get_path(file_name, output=True), pts)




get_points("calibration_points3.txt")

store_click(get_path("left.jpg"), "left.txt")
store_click(get_path("right.jpg"), "right.txt")