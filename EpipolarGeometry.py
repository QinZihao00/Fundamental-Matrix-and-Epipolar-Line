# QinZihao From NTU, Singapore
import numpy as np
import matplotlib.pyplot as plt


def convert_coordinate(points, img_size):
    """
    :param points:  List of tuples [(x1, y1), ...]
    :param img_size:      tuple (x, y)
    :return: List of tuples [(u1, v1), ...]
    """
    x_half, y_half = img_size[0]//2, img_size[1]//2
    result = [(x-x_half, y_half-y) for x, y in points]
    return result


def get_FundamentalMatrix(left_points, right_points):
    """
    params: left_points:  List of tuples [(u1, v1), ...]
    params: right_points: List of tuples [(u1', v1'), ...]
    return: F:            3 * 3 numpy ndarray - Fundamental Matrix
    Use eight pairs of points from left and right images to compute the fundamental matrix.
    """
    assert len(left_points) == len(right_points) == 8
    A = []
    for idx in range(len(left_points)):
        u, v = left_points[idx]
        u_, v_ = right_points[idx]
        eqn_param = [u * u_, u * v_, u, v * u_, v * v_, v, u_, v_]
        A.append(eqn_param)
    A = np.array(A).astype(np.float64)
    B = np.array([-1]*8).astype(np.float64)
    F = np.append(np.linalg.solve(A, B), 1.).reshape(3, 3)
    return F


def get_epipolar_param(F, point, point_image=None):
    """
    params: F:           3 * 3 numpy ndarray - Fundamental Matrix
    params: point:       Tuple - point coordinates
    params: point_image: Str - 'left' or 'right'
    return: w, b:        Floating - Epipolar line equation y = w * x + b parameters
    Compute the epipolar line equation on the corresponding image.
    """
    point = np.append(np.array(point), 1.0)    # (u, v, 1)
    
    if point_image == 'left':    # Input left point -> find epipolar line in right image
        a,b,c = np.matmul(point, F)
        return -a/b, -c/b
    if point_image == 'right':   # Input right point -> find epipolar line in left image
        a,b,c = np.matmul(point, np.transpose(F))
        return -a/b, -c/b


def plot_epipolar_line(w, b, image_path, save_path):
    """
    params: w: Float / list of float
    params: b: Float / list of float
    params: image_path: Str
    params: save_path: Str
    return: None
    Show image with epipolar line and save new image.
    """
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(img.shape[1]/100., img.shape[0]/100.))
    ax.imshow(img, extent=[-img.shape[1]/2., img.shape[1]/2., -img.shape[0]/2., img.shape[0]/2.])
    for _, (cur_w, cur_b) in enumerate(zip(w, b)):
        x = np.linspace(-img.shape[1]/2, img.shape[1]/2)
        line_eqn = cur_w * x + cur_b
        ax.plot(x, line_eqn, linewidth=8)
    ax.axis('off')
    fig.savefig(save_path, dpi=100)
    plt.show()


if __name__ == "__main__":
    # Give your eight pairs of points to solve for F
    F = get_FundamentalMatrix(...)
    # Choose a point to get epipolar line equation parameters
    w, b = get_epipolar_param(...)
    # Plot epipolar line on the corresponding image and save as figure
    plot_epipolar_line(...)

