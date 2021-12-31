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
import scipy
import os
from pysteps.visualization import motion_plot, quiver, streamplot
import math
from sklearn.preprocessing import normalize
#from wavepy.surface_from_grad import frankotchellappa
from frankoChellappa import frankotchellappa
import plotly.graph_objects as go
import numpy as np

def generate_mask(img):
    tresh = 0.1
    img = normarlize_percentile(img, 99)
    mask = np.zeros_like(img[0])
    for i in range(len(img)):
        I = img[i]
        I[I < tresh] = 0
        I[I >= tresh] = 1
        mask += I

    mask[mask > 0] = 1
    """plt.imshow(mask)
    plt.show()"""
    return mask


def normarlize_percentile(img, percentile):
    normalized_factor = np.percentile(img, percentile)
    return img / normalized_factor


def get_points(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        points = np.loadtxt(lines)
    return points


def get_img(file_path, nbr):
    res = []
    for i in range(nbr):
        # reading the image
        if i < 9:
            im = cv2.imread(file_path + "Image_" + str(0) + str(i + 1) + ".png", cv2.IMREAD_GRAYSCALE).astype(float)
        else:
            im = cv2.imread(file_path + "Image_" + str(i + 1) + ".png", cv2.IMREAD_GRAYSCALE).astype(float)
        res.append(im)
    return res


def compute_surfNorm(I, L, mask):
    '''compute the surface normal vector'''
    shape = [I.shape[1], I.shape[2]]
    I = np.array(I).reshape((20, 1, -1)).squeeze(1)
    L = np.array(L).T
    N = np.linalg.lstsq(L, I, rcond=-1)[0].T
    N = normalize(N, axis=1)
    return N


def photometric_stereo(I, L, mask):
    I = I * mask
    N = compute_surfNorm(I, L, mask)
    n = np.reshape(N.copy(), (I.shape[1], I.shape[2], 3))
    p = np.zeros_like(n[:, :, 1])
    q = np.zeros_like(n[:, :, 1])
    cond = n[:, :, 2]
    """p[cond == 0] = 0
    q[cond == 0] = 0"""
    p = n[:, :, 0] / -n[:, :, 2]
    q = n[:, :, 1] / -n[:, :, 2]

    p[np.isnan(p)] = 0
    q[np.isnan(q)] = 0

    """ p = n[:, :, 0] / -n[:, :, 2]
    q = n[:, :, 1] / -n[:, :, 2]"""
    return p, q


def show_surf_norm(p, q):
    surf_norm = np.zeros([p.shape[0], p.shape[1], 3])
    surf_norm[:, :, 0] = p
    surf_norm[:, :, 1] = q
    surf_norm[:, :, 2] = -1
    plt.imshow(surf_norm)
    plt.show()


def compute_depth(mask, N):
    """
    compute the depth picture
    """
    im_h, im_w = mask.shape
    N = np.reshape(N, (im_h, im_w, 3))

    # =================get the non-zero index of mask=================
    obj_h, obj_w = np.where(mask != 0)
    no_pix = np.size(obj_h)  # 37244
    full2obj = np.zeros((im_h, im_w))
    for idx in range(np.size(obj_h)):
        full2obj[obj_h[idx], obj_w[idx]] = idx

    M = scipy.sparse.lil_matrix((2 * no_pix, no_pix))
    v = np.zeros((2 * no_pix, 1))

    # ================= fill the M&V =================
    for idx in range(no_pix):
        # obtain the 2D coordinate
        h = obj_h[idx]
        w = obj_w[idx]
        # obtian the surface normal vector
        n_x = N[h, w, 0]
        n_y = N[h, w, 1]
        n_z = N[h, w, 2]

        row_idx = idx * 2
        if mask[h, w + 1]:
            idx_horiz = full2obj[h, w + 1]
            M[row_idx, idx] = -1
            M[row_idx, idx_horiz] = 1
            if n_z == 0:
                v[row_idx] = 0
            else:
                v[row_idx] = -n_x / n_z
        elif mask[h, w - 1]:
            idx_horiz = full2obj[h, w - 1]
            M[row_idx, idx_horiz] = -1
            M[row_idx, idx] = 1
            if n_z == 0:
                v[row_idx] = 0
            else:
                v[row_idx] = -n_x / n_z

        row_idx = idx * 2 + 1
        if mask[h + 1, w]:
            idx_vert = full2obj[h + 1, w]
            M[row_idx, idx] = 1
            M[row_idx, idx_vert] = -1
            if n_z == 0:
                v[row_idx] = 0
            else:
                v[row_idx] = -n_y / n_z
        elif mask[h - 1, w]:
            idx_vert = full2obj[h - 1, w]
            M[row_idx, idx_vert] = 1
            M[row_idx, idx] = -1
            if n_z == 0:
                v[row_idx] = 0
            else:
                v[row_idx] = -n_y / n_z

    # =================sloving the linear equations Mz = v=================
    MtM = M.T @ M
    Mtv = M.T @ v
    z = scipy.sparse.linalg.spsolve(MtM, Mtv)

    std_z = np.std(z, ddof=1)
    mean_z = np.mean(z)
    z_zscore = (z - mean_z) / std_z
    outlier_ind = np.abs(z_zscore) > 10
    z_min = np.min(z[~outlier_ind])
    z_max = np.max(z[~outlier_ind])

    Z = mask.astype('float')
    for idx in range(no_pix):
        # obtain the position in 2D picture
        h = obj_h[idx]
        w = obj_w[idx]
        Z[h, w] = (z[idx] - z_min) / (z_max - z_min) * 255

    depth = Z
    return depth


if __name__ == "__main__":
    Name = 'turtle'
    nbr_img = 20
    object_path = 'Input/PSData/PSData/' + Name + '/Objects/'
    light_path = 'Input/PSData/PSData/' + Name + '/light_directions.txt'

    img = get_img(object_path, nbr_img)
    I = normarlize_percentile(img, 99)
    mask = generate_mask(img)
    L = get_points(light_path)

    n = compute_surfNorm(I, L, mask)
    p, q = photometric_stereo(I, L, mask)
    show_surf_norm(p, q)


    z = frankotchellappa(p, q)

    # z = compute_depth(mask, n)

    depth = z
    plt.imshow(z)
    plt.show()

    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np


    z = depth.reshape(-1,1)
    x = np.linspace(0,z.shape[0])
    y = x.copy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()
    
    import plotly.graph_objects as go
    import numpy as np

    x = np.outer(np.linspace(0, I.shape[2],1 ), np.ones_like(I))
    y = x.copy().T
    z = depth

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

    fig.show()
    """











