from gurobipy import *
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


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

    QDCC = quad2[0]
    qDCC = quad2[1]
    rhoDCC = quad2[2]

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


def vis_ellipsoid(ellipsoid,cone,DC,point):
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

    x_org = x*v1[0,0]+y*v2[0,0]+z*v3[0,0]
    y_org = x*v1[1,0]+y*v2[1,0]+z*v3[1,0]
    z_org = x*v1[2,0]+y*v2[2,0]+z*v3[2,0]

    # Plot:
    ax.plot_surface(x_org, y_org, z_org,  rstride=4, cstride=4, color='b')

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    ############ cone
    # x^T Q x + 2 q^T x + rho <=0
    # x^T H Lambda H^T x + 2q^T x +rho <=0
    # x^T H Lambda H^T x + 2q^T H H^T x +rho <=0
    # y^T Lambda y + 2 q^T H y +rho<=0
    # y^T Lambda y + 2 s^T y +rho<=0
    # lambda_2*y_2**2+lambda_3*y_3**3+2*s_2*y_2+2*s_3*y_3+rho<=-lambda_1*y_1**2-2*s_1*y_1
    # lambda_2*(y_2**2+2*s_2/lambda_2 y_2+(s_2/lambda_2)**2)
    #+lambda_3*(y_3**2+2*s_3/lambda_3 y_3+(s_3/lambda_3)**2)
    #+(rho-s_2**2/lambda_2-s_3**2/lambda_3-s_1**2/lambda_1)
    #<=-lambda_1*(y_1**2+2*s_1/lambda_1*y_1+(s_1/lambda_1)**2)

    # lambda_2*(y_2+(s_2/lambda_2))**2
    #+lambda_3*(y_3+(s_3/lambda_3))**2
    #<=-lambda_1*(y_1+(s_1/lambda_1))**2
    QDCC = cone[0]
    qDCC = cone[1]
    rhoDCC = cone[2]

    [eig_val_vec,eig_vec_mat] = np.linalg.eig(QDCC)
    v1 = eig_vec_mat[:,0]
    v2 = eig_vec_mat[:,1]
    v3 = np.transpose(np.cross(np.transpose(v1),np.transpose(v2)))
    eig_vec_mat[:,2] = v3

    s_vec = np.transpose(eig_vec_mat).dot(qDCC)


    r2 = np.sqrt(eig_val_vec[1]/(-eig_val_vec[0]))
    r3 = np.sqrt(eig_val_vec[2]/(-eig_val_vec[0]))

    shift_x = -s_vec[0]/eig_val_vec[0]
    shift_y = -s_vec[1]/eig_val_vec[1]
    shift_z = -s_vec[2]/eig_val_vec[2]



    # Set up the grid in polar
    theta = np.linspace(0,2*np.pi,50)
    r = np.linspace(0,50,50)
    T, R = np.meshgrid(theta, r)

    # Then calculate X, Y, and Z
    Y = R * np.cos(T)
    Z = R * np.sin(T)
    X = np.sqrt(Y**2 + Z**2)
    X2 = -X

    Y = Y/r2
    Z = Z/r3

    X = X+shift_x
    X2 = X2+shift_x
    Y = Y+shift_y
    Z = Z+shift_z

    X_org = X*v1[0,0]+Y*v2[0,0]+Z*v3[0,0]
    Y_org = X*v1[1,0]+Y*v2[1,0]+Z*v3[1,0]
    Z_org = X*v1[2,0]+Y*v2[2,0]+Z*v3[2,0]

    X2_org = X2*v1[0,0]+Y*v2[0,0]+Z*v3[0,0]
    Y2_org = X2*v1[1,0]+Y*v2[1,0]+Z*v3[1,0]
    Z2_org = X2*v1[2,0]+Y*v2[2,0]+Z*v3[2,0]


    ax.plot_wireframe(X_org, Y_org, Z_org)
    ax.plot_wireframe(X2_org, Y2_org, Z2_org)

    # disjunctive
    a = DC[0]
    alpha = DC[1]
    beta = DC[2]

    xx, yy = np.meshgrid(np.linspace(-10.0,10.0,10), np.linspace(-10.0,10.0,10))

    # calculate corresponding z
    z1 = (-a[0,0] * xx - a[1,0] * yy + alpha) * 1. /a[2,0]
    z2 = (-a[0,0] * xx - a[1,0] * yy + beta) * 1. /a[2,0]

    # plot the surface
    f1 = ax.plot_surface(xx, yy, z1)
    f2 = ax.plot_surface(xx, yy, z2)

    coeff = 0.5
    f1.set_facecolor((0, 0, 1, coeff))
    f2.set_facecolor((0, 0, 1, coeff))




    plt.plot([point[0,0]],[point[1,0]],[point[2,0]],'ro')



    plt.show()
