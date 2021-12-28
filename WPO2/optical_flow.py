# Import required modules
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as filter2
import os
from pysteps.visualization import motion_plot, quiver, streamplot
from view_flow import flow_uv_to_colors

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


"""def compute_grad(im1, im2):

    return I_x, I_y, I_t"""


def show_image(name, image):
    if image is None:
        return

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#compute magnitude in each 8 pixels. return magnitude average
def get_magnitude(u, v):
    scale = 3
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg



def draw_quiver(u,v,beforeImg):
    scale = 1
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 3):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')

    plt.draw()
    plt.show()



#compute derivatives of the image intensity values along the x, y, time
def get_derivatives(img1, img2):
    #derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1, x_kernel) + filter2(img2, x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx, fy, ft]



#input: images name, smoothing parameter, tolerance
#output: images variations (flow vectors u, v)
#calculates u,v vectors and draw quiver



def computeHS(img1, img2, lamda, epsilon, max_it):

    beforeImg = img1
    afterImg = img2

    #removing noise
    beforeImg = cv2.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv2.GaussianBlur(afterImg, (5, 5), 0)

    # set up initial values
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    """avg_kernel =(1/9)* np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], float)"""
    """avg_kernel =(1/4)* np.array([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], float)"""
    iter_counter = 0
    diff = 1
    while diff > epsilon and iter_counter < max_it:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = lamda ** 2 + fx ** 2 + fy ** 2
        #d = 1 + lamda * (fx ** 2 + fy ** 2)
        prev_u = u
        prev_v = v

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.sum(np.square(u-prev_u) + np.square(v -prev_v))/len(u)
        print(iter_counter, diff)



    return [u, v]


if __name__ == "__main__":
    img = []
    for i in range(8):
        # reading the image
        im = cv2.imread(get_path("frame"+str(i+1)+".png"), cv2.IMREAD_GRAYSCALE).astype(float)
        img.append(im)
    u, v = computeHS(img[0], img[2], lamda=15, epsilon=10 ** -5, max_it=1000)
    # draw_quiver(u, v, beforeImg)

    """ax = plt.figure().gca()
    ax.imshow(img[0], cmap='gray')"""
    """y, x = np.mgrid[0:u.shape[0],0:u.shape[1]]
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=25, pivot='mid',color='r')"""
    #motion_plot(np.array([u, v]), step=10)

    flow = flow_uv_to_colors(u,v)
    plt.imshow(flow)
    plt.show()

    draw_quiver(u,v,img[0])



