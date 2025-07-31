import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_3dPoints(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = points.T

    ax.scatter(points[0], points[2], points[1], c='b', marker='o')

    plt.show()