from gurobipy import *
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def contour_quad(quad,obj_coeff,quad2,lin,x_bound,y_bound):

    Q = quad[0]
    q = quad[1]
    rho = quad[2]

    print('+++++++++++++',Q,q,rho)

    QDCC = quad2[0]
    qDCC = quad2[1]
    rhoDCC = quad2[2]

    print('pppppppppppp',QDCC,qDCC,rhoDCC)

    a = lin[0]
    alpha = lin[1]
    beta = lin[2]

    Nx = 500
    Ny = 500

    x_lb = x_bound[0]
    x_ub = x_bound[1]

    y_lb = y_bound[0]
    y_ub = y_bound[1]

    x = np.linspace(x_lb,x_ub,Nx)
    y = np.linspace(y_lb,y_ub,Ny)

    X, Y = np.meshgrid(x, y)

    Z = np.zeros((Nx,Ny))
    Z2 = np.zeros((Nx,Ny))

    for i in range(Nx):
        for j in range(Ny):

            loc_x = np.matrix([[x[i]],[y[j]]])
            loc_z = np.transpose(loc_x).dot(Q).dot(loc_x)\
            +2.0*np.transpose(q).dot(loc_x)+rho

            loc_z2 = np.transpose(loc_x).dot(QDCC).dot(loc_x)\
            +2.0*np.transpose(qDCC).dot(loc_x)+rhoDCC

            Z[j,i] =loc_z
            Z2[j,i] =loc_z2


    x_a1 = x_lb
    y_a1 = ((alpha-a[0]*x_a1)/a[1])[0,0]
    x_a2 = x_ub
    y_a2 = ((alpha-a[0]*x_a2)/a[1])[0,0]

    x_b1 = x_lb
    y_b1 = ((beta-a[0]*x_b1)/a[1])[0,0]
    x_b2 = x_ub
    y_b2 = ((beta-a[0]*x_b2)/a[1])[0,0]

    x_obj1 = x_lb
    y_obj1 = ((0.0-obj_coeff[0]*x_obj1)/obj_coeff[1])[0,0]
    x_obj2 = 0.0
    y_obj2 = 0.0

    CS1 = plt.contour(X, Y, Z)
    CS2 = plt.contour(X, Y, Z2)
    plt.plot([x_a1,x_a2],[y_a1,y_a2],'r')
    plt.plot([x_b1,x_b2],[y_b1,y_b2],'b')
    plt.plot([x_obj1,y_obj1],[x_obj2,y_obj2],'--b')
    plt.clabel(CS1, inline=1, fontsize=10)
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.show()
