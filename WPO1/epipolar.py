import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from calibration import get_path, get_points, compute_m, get_2D_from_3D, plot
import matplotlib.image as img
#  Global variable
points = []
points_3d = []


def click_event(event, x, y):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def store_click(image, file_name):
    global points
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

    points = np.array(points)
    np.savetxt(get_path(file_name, output=True, overwritte=True), points)


def make_svd_decomposition(a):
    u, s, v = np.linalg.svd(a)
    v = v[-1, :]
    #  v = np.reshape(np.array(v), (3, 4))
    v = v[0:3]/v[-1]
    return v


def compute_a(mr, ml, xr, xl):
    global points_3d
    ur, vr = xr[:, 0], xr[:, 1]
    ul, vl = xl[:, 0], xl[:, 1]
    for i in range(xr.shape[0]):
        a = [ur[i]*mr[2, :]-mr[0, :], vr[i]*mr[2, :]-mr[1, :],
             ul[i] * ml[2, :] - ml[0, :], vl[i] * ml[2, :] - ml[1, :]]
        a = make_svd_decomposition(a)
        points_3d.append(a)


def calibration(NAME, screen=False):
    a = get_points("calibration_points3.txt")
    #  store_click(get_path(NAME+".jpg"), NAME+".txt")
    x = get_points(NAME+".txt", output=True)
    m = compute_m(a, x[0:12])
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
    axes = [axe_x, axe_y, axe_z]
    if screen:
        plot(image, pts_refined, origin, axes)

    return x, m



def reconstruction(first=False, show_calibration=False):
    global points
    NAME = "right"
    points = []
    if first:
        store_click(get_path(NAME + ".jpg"), NAME + ".txt")
    xr, mr = calibration(NAME, show_calibration)
    NAME = "left"
    points = []
    if first:
        store_click(get_path(NAME + ".jpg"), NAME + ".txt")
    xl, ml = calibration(NAME, show_calibration)
    compute_a(mr, ml, xr, xl)
    return ml, mr


def plot_cube(index_i):
    Z = np.zeros([8, 3])
    Z[0:7, :] = np.array(points_3d[index_i:index_i + 7])
    Z[7, :] = np.array([Z[6, 0], Z[4, 1], Z[5, 2]])

    # list of sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[7], Z[4]],
             [Z[2], Z[3], Z[5], Z[6]],
             [Z[1], Z[2], Z[6], Z[7]],
             [Z[4], Z[7], Z[1], Z[0]]]
    return verts


def plot_structure():
    a =points_3d
    Z = np.zeros([6, 3])
    Z[0, :] = np.array([0, 0, 0])
    Z[1, :] = np.array([0, 0, points_3d[11, 2]])
    Z[2, :] = np.array(points_3d[10])
    Z[3, :] = np.array([0, points_3d[6][1], 0])
    Z[4, :] = np.array(points_3d[5])
    Z[5, :] = np.array([points_3d[1][0], 0, 0])

    # list of sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[0], Z[1], Z[4], Z[5]]]
    return verts


def plot_triangle(index_i):
    Z = np.zeros([4, 3])
    Z[0:4, :] = np.array(points_3d[index_i:index_i + 4])

    # list of sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2]],
             [Z[0], Z[1], Z[3]],
             [Z[3], Z[0], Z[2]],
             [Z[0], Z[2], Z[3]]]
    return verts


def plot_3d():
    global points_3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot vertices
    '''ax.scatter3D((points_3d[i, :]) for i in range(np.array(points_3d).shape(0)))'''
    points_3d = np.array(points_3d)
    ax.scatter3D(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    '''ax.text3D(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], '%s' % (str(np.arange(1, points_3d.shape[0]))), size=20, zorder=1,
            color='k')'''
    # plot sides
    ax.add_collection3d(Poly3DCollection(plot_cube(12),
                                         facecolors='#929591', linewidths=1, edgecolors='k', alpha=.25))
    ax.add_collection3d(Poly3DCollection(plot_cube(23),
                                         facecolors='#af884a', linewidths=1, edgecolors='k', alpha=.25))
    ax.add_collection3d(Poly3DCollection(plot_triangle(19),
                                         facecolors='#e50000', linewidths=1, edgecolors='k', alpha=.25))

    ax.add_collection3d(Poly3DCollection(plot_structure(),
                                         facecolors='#929591', linewidths=1, edgecolors='k', alpha=.25))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xmin, xmax = points_3d[:, 0].min(), points_3d[:, 0].max()
    ymin, ymax = points_3d[:, 1].min(), points_3d[:, 1].max()
    zmin, zmax = points_3d[:, 2].min(), points_3d[:, 2].max()

    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)
    """azim 
    if azim is not None:
        ax.azim = azim
    if dist is not None:
        ax.dist = dist
    if elev is not None:
        ax.elev = elev"""
    plt.show()


def mse(result, target):
    res = 0
    n = len(target)
    for i in range(n):
        res += abs(target[i]-result[i])/n
    return res


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

def plot (image, pts, origin, axes):
    im = img.imread(image)
    plt.imshow(im)
    pts = np.array(pts)
    axes = np.array(axes)
    origin = np.array(origin)
    plt.scatter(x=pts[:, 0], y=pts[:, 1])
    for i in range(axes.shape[0]):
        axe_x = np.array([origin[i][0], axes[i][0]])
        axe_y = np.array([origin[i][1], axes[i][1]])
        plt.plot(axe_x, axe_y)
    plt.show()


def is_paralelle (d_in, d_out):
    temp = np.zeros([3,2])
    for i in range(d_in.shape[0]):
        temp[i, :] = d_out[i]-d_in[i]
    rank = np.linalg.matrix_rank(temp)
    if rank >= 1:
        return False
    else:
        return True

def auto_calibration(NAME, screen=True):
    a = get_points("calibration_points3.txt")
    #  store_click(get_path(NAME+".jpg"), NAME+".txt")
    x = get_points(NAME+".txt", output=True)
    m = compute_m(a, x[0:12])
    # re-find axes and points
    factor = 2
    par = [factor*100, factor*100, factor*100]
    spacing = 40

    d1_in = get_2D_from_3D(m, [-par[0], par[0], spacing, 1])
    d1_end = get_2D_from_3D(m, [par[0], -par[0], spacing, 1])

    d2_in = get_2D_from_3D(m, [-par[1], par[1], 2*spacing, 1])
    d2_end = get_2D_from_3D(m, [par[1], -par[1], 2*spacing, 1])

    d3_in = get_2D_from_3D(m, [-par[2], par[2], 3*spacing, 1])
    d3_end = get_2D_from_3D(m, [par[2], -par[2], 3*spacing, 1])
    pts_refined = []
    for i in range(a.shape[0]):
        pts_refined.append(get_2D_from_3D(m, [a[i, 0], a[i, 1], a[i, 2], 1]))
    np.savetxt(get_path(NAME+"_reconstruction.txt", output=True, overwritte=True), pts_refined)
    image = get_path(NAME+".jpg")
    d_in = np.array([d1_in, d2_in, d3_in])
    d_end = np.array([d1_end, d2_end, d3_end])

    if screen:
        plot(image, pts_refined, d_in, d_end)
    is_paralelle(d_in, d_end)
    return x, m
if __name__ == "__main__":
    # set first = True to re-take the points manually
    NAME = "right"
    x, m = auto_calibration(NAME)
    #T, R, A, K, M = get_param(m)
    """first = False
    # set show_calibration = True to visualize the axes resulting from calibration
    show_calibration = False
    ml, mr = reconstruction(first, show_calibration)
    plot_3d()
    # Values of interest
    print("MSE (x, y, z) = ", mse(points_3d, get_points("calibration_points3.txt")))
    print("\n---------------LEFT_PARAM------------------\n")
    get_param(ml)
    print("\n---------------RIGHT_PARAM------------------\n")
    get_param(mr)"""


