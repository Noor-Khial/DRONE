import numpy as np
from scipy.special import binom

def bernstein_poly(i, n, t):
    return binom(n, i) * ( t**(n-i) ) * ( (1 - t)**i )

## n_points is the number of epochs in one episode.
def bezier_curve(points, n_points=200):
    n = len(points) - 1
    x_vals = np.array([p[0] for p in points])
    y_vals = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, n_points)
    polynomial_array = np.array([bernstein_poly(i, n, t) for i in range(0, n+1)])
    xvals = np.dot(x_vals, polynomial_array)
    yvals = np.dot(y_vals, polynomial_array)
    return xvals[::-1], yvals[::-1]

if __name__ == "__main__":
    control_points = [(45, 38), (55, 55)]
    xvals, yvals = bezier_curve(control_points)
    print(xvals, yvals)
