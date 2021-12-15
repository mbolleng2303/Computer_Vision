# Import required modules
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as img
#  Global variable
pts = []


def get_path(name, output=False, overwritte = False):
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
            if overwritte:
                os.remove(file_dir)
            return file_dir
        else:
            return file_dir


def get_points(text_file, output=False):
    file_path = get_path(text_file, output)
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
    np.savetxt(get_path(file_name, output=True, overwritte=True), pts)


def compute_m(a, x):
    u = x[:, 0]
    v = x[:, 1]
    m = np.zeros([2*x.shape[0], a.shape[0]])
    for i in range(x.shape[0]):
        m[2*i, :] = [a[i, 0], a[i, 1], a[i, 2], 1, 0, 0, 0, 0, -u[i]*a[i, 0], -u[i]*a[i, 1], -u[i]*a[i, 2], -u[i]]
        m[2*i+1, :] = [0, 0, 0, 0, a[i, 0], a[i, 1], a[i, 2], 1, -v[i]*a[i, 0], -v[i]*a[i, 1], -v[i]*a[i, 2], -v[i]]
    m = make_svd_decomposition(m)
    return m


def make_svd_decomposition(m):
    u, s, v = np.linalg.svd(m)
    v = v[-1, :]
    v = np.reshape(np.array(v), (3, 4))
    v = v/v[-1, -1]
    return v


def get_2D_from_3D(m, axis):
    axis = np.array(axis).T
    proj = np.matmul(m, axis)
    homogenous = proj[2]
    x = proj[0]/homogenous
    y = proj[1]/homogenous
    return [x, y]


def plot (image, pts, origin, axes):
    im = img.imread(image)
    plt.imshow(im)
    pts = np.array(pts)
    axes = np.array(axes)
    plt.scatter(x=pts[:, 0], y=pts[:, 1])
    axe_x = np.array([origin[0], axes[0][0], origin[0], axes[1][0], origin[0], axes[2][0]])
    axe_y = np.array([origin[1], axes[0][1], origin[1], axes[1][1], origin[1], axes[2][1]])
    plt.plot(axe_x, axe_y)
    plt.show()


def calibration(NAME):
    a = get_points("calibration_points3.txt")
    #store_click(get_path(NAME+".jpg"), NAME+".txt")
    x = get_points(NAME+".txt", output=True)
    x = x[0:12]
    m = compute_m(a, x)
    # re-find axes and points
    axe_length = 70
    origin = get_2D_from_3D(m, [0, 0, 0, 1])
    axe_x = get_2D_from_3D(m, [axe_length, 0, 0, 1])
    axe_y = get_2D_from_3D(m, [0, axe_length,  0, 1])
    axe_z = get_2D_from_3D(m, [0, 0, axe_length, 1])
    pts_refined = []
    for i in range(a.shape[0]):
        pts_refined.append(get_2D_from_3D(m, [a[i, 0], a[i, 1], a[i, 2], 1]))
    np.savetxt(get_path(NAME+"_reconstruction.txt", output=True, overwritte=True), pts_refined)
    image = get_path(NAME+".jpg")
    plot(image, pts_refined, origin, [axe_x, axe_y, axe_z])
    return m


def get_param(m):
    m1, m2, m3 = m[0, 0:3], m[1, 0:3], m[2, 0:3]

    cx, cy = np.dot(m1, m3), np.dot(m2, m3)
    fx, fy = np.linalg.norm(np.cross(m1, m3)), np.linalg.norm(np.cross(m2, m3))
    r1, r2, r3 = (m1 - cx * m3) / fx, (m2 - cy * m3) / fy, m3
    tx, ty, tz = (m[0, 3] - cx * m[2, 3]) / fx, (m[1, 3] - cy * m[2, 3]) / fy, m[2, 3]

    T = np.array([tx, ty, tz])
    R = np.array([r1, r2, r3])
    A = np.array([[R[0][0], R[0][1], R[0][2], T[0]],
                  [R[1][0], R[1][1], R[1][2], T[1]],
                  [R[2][0], R[2][1], R[2][2], T[2]],
                  [0, 0, 0, 1]])

    s = 0
    intrinsics = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])
    projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    K = np.matmul(intrinsics, projection)
    M = np.matmul(K, A)

    assert M.all() == m.all(), "problem !"
    print("T=", T)
    print("\n")
    print("R=", R)
    print("\n")
    print("A=", A)
    print("\n")
    print("K=", K)
    print("\n")
    print("m=", m, "\n M= \n", M)
    return T, R, A, K, M


if __name__ == "__main__":
    NAME = "right"
    m = calibration(NAME)
    T, R, A, K, M = get_param(m)




