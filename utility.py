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


def vis_ellipsoid(ellipsoid):
# x^T Q x + 2 q^T x +rho
# = x^T H Lambda H^T x + 2 q^T H H^Tx + rho
# = y^T Lambda y + 2 q^T H y+rho
# = y^T Lambda y + 2 s^T y+rho
# = lambda1*y1**2+lambda2*y2**2+lambda3*y3**2+2*s1*y1+2*s2*y2+2*s3*y3+rho
# = lambda1*(y1**2+2*s1/lambda1*y1+(s1/lambda1)**2)
# +lambda2*(y2**2+2*s2/lambda2*y2+(s2/lambda2)**2)
# +lambda3*(y3**2+2*s3/lambda3*y3+(s3/lambda3)**2)
# +(rho-s1**2/lambda1-s2**2/lambda2-s3**2/lambda3)=0
#
# lambda1*(y1**2+2*s1/lambda1*y1+(s1/lambda1)**2)
# +lambda2*(y2**2+2*s2/lambda2*y2+(s2/lambda2)**2)
# +lambda3*(y3**2+2*s3/lambda3*y3+(s3/lambda3)**2)=
# -(rho-s1**2/lambda1-s2**2/lambda2-s3**2/lambda3):=rho_norm
#
# lambda1/rho_norm*(y1+(s1/lambda1))**2
# +lambda2/rho_norm*(y2+(s2/lambda2))**2
# +lambda2/rho_norm*(y3+(s3/lambda3))**2=1




    # input info
    Q = ellipsoid[0]
    q = ellipsoid[1]
    rho = ellipsoid[2]

    # get the principal direction and 1/radius
    [Lambda,axis_dir] = np.linalg.eig(Q)
    v1 = axis_dir[:,0]
    v2 = axis_dir[:,1]
    v3 = np.transpose(np.cross(np.transpose(v1),np.transpose(v2)))
    axis_dir[:,2] = v3

    s_vec = np.transpose(axis_dir).dot(q)
    rho_norm = -(rho-s_vec[0]**2/Lambda[0]-s_vec[1]**2/Lambda[1]-s_vec[2]**2/Lambda[2])
    print('s_vec,rho_norm',s_vec,rho_norm)

    r1 = Lambda[0]/rho_norm
    r2 = Lambda[1]/rho_norm
    r3 = Lambda[2]/rho_norm

    shift_x = -s_vec[0]/Lambda[0]
    shift_y = -s_vec[1]/Lambda[1]
    shift_z = -s_vec[2]/Lambda[2]



    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    coefs = (r1, r2, r3)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    x = x+shift_x
    y = y+shift_y
    z = z+shift_z

    print(';;;;;;shift_x',shift_x,shift_y,shift_z)

    x_org = x*v1[0,0]+y*v2[0,0]+z*v3[0,0]
    y_org = x*v1[1,0]+y*v2[1,0]+z*v3[1,0]
    z_org = x*v1[2,0]+y*v2[2,0]+z*v3[2,0]

    x_test_vec = np.matrix([[x_org[50,50]],[y_org[50,50]],[z_org[50,50]]])
    print('test!!!',np.transpose(x_test_vec).dot(Q).dot(x_test_vec)+2.0*np.transpose(q).dot(x_test_vec)+rho)
    x_test_vec = np.matrix([[x_org[10,10]],[y_org[10,10]],[z_org[10,10]]])
    print('test!!!',np.transpose(x_test_vec).dot(Q).dot(x_test_vec)+2.0*np.transpose(q).dot(x_test_vec)+rho)
    x_test_vec = np.matrix([[x_org[0,0]],[y_org[0,0]],[z_org[0,0]]])
    print('test!!!',np.transpose(x_test_vec).dot(Q).dot(x_test_vec)+2.0*np.transpose(q).dot(x_test_vec)+rho)

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='r')
    ax.plot_surface(x_org, y_org, z_org,  rstride=4, cstride=4, color='b')

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    plt.show()
