from gurobipy import *
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import time

import utility as ut

# sicheng???
class SOCP_DCC:

    # the constraints are:
    # Ax=b
    # x^TJx<=0

    def __init__(self,A,b,J,c,int_index,lb_ub):

        # quad constraint

        self.H = ut.nullspace(A)

        x = np.linalg.lstsq(A, b)
        self.w_sol = x[0]

        self.Q = np.transpose(self.H).dot(J).dot(self.H)
        self.q = np.transpose(np.transpose(self.w_sol).dot(J).dot(self.H))
        self.rho = np.transpose(self.w_sol).dot(J).dot(self.w_sol)

        # cut constraint

        self.a = np.transpose(self.H[int_index,:])

        lb = lb_ub[0]
        ub = lb_ub[1]

        self.alpha = (ub-self.w_sol[int_index])[0,0]
        self.beta = (lb-self.w_sol[int_index])[0,0]

        self.pro_size_orig = len(A[0,:])
        self.pro_size = len(self.q)

        # obj

        self.obj_coeff = np.transpose(np.transpose(c).dot(self.H))





    def Nullspace_DCC(self):

        constr_group = [self.Q,self.q,self.rho]
        disjunction_group = [self.a,self.alpha,self.beta]

        m = Model("qcp")

        x = (np.zeros(self.pro_size)).tolist()
        for i in range(self.pro_size):
            x[i] = m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name='x%i' %i)

        m.update()

        obj = 0.0
        for i in range(self.pro_size):
            obj += self.obj_coeff[i]*x[i]
        m.setObjective(obj, GRB.MAXIMIZE)

        sum_x = 0.0
        for i in range(self.pro_size):
            for j in range(self.pro_size):
                sum_x += x[i]*self.Q[i,j]*x[j]
        for i in range(self.pro_size):
            sum_x += 2.0*self.q[i]*x[i]
        sum_x += self.rho

        m.addQConstr(sum_x<=0.0)

        m.optimize()
        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))

        opt_modelwithN = [m,self.pro_size]

        DCC_hull = DCC_class(opt_modelwithN,constr_group,disjunction_group)

        DCC_hull.update_constr()

        DCC_hull.opt_model.optimize()


        for v in DCC_hull.opt_model.getVars():
            print('%s %g' % (v.varName, v.x))

        x_new = np.zeros((self.pro_size,1))
        i = 0

        var_all_new = DCC_hull.opt_model.getVars()
        for i in range(self.pro_size):
            x_new[i,0] =var_all_new[i].x
            i += 1

        x_original = self.w_sol+self.H.dot(x_new)

        print('00000000000000',x_original)




        # ut.contour_quad([self.Q,self.q,self.rho],\
        # self.obj_coeff,\
        # [DCC_hull.QDCC,DCC_hull.qDCC,DCC_hull.rhoDCC],\
        # [self.a,self.alpha,self.beta],\
        # [-5.0,5.0],[-5.0,5.0])















class DCC_class:

    # @Sicheng He, May 11, 2016

    # disjunctive conic cuts for (mixed integer) second order conic programming (MISOCP)
    # inputs: (i).    the model and number of original variables
    #         (ii).   the (part of) constraints of a optimization problem ((Q,q,rho) group);
    #         (iii).  the disjunction ((a,alpha,beta) group)
    # outputs:
    #         new model with DCC cuts (with auxiliary variables though)
    # based on:
    #         "A conic representation of the convex hull of disjunctive sets and conic cuts
    #          for integer second order cone optimization"
    #          Authors: Belloti/Goez/Polik/Ralphs/Terlaky

    def __init__(self,opt_modelwithN,constr_group,disjunction_group):

        self.opt_model = opt_modelwithN[0]
        self.opt_model_N = opt_modelwithN[1]

        self.Q = constr_group[0]
        self.q = constr_group[1]
        self.rho = constr_group[2]

        self.a = disjunction_group[0]
        self.alpha = disjunction_group[1]
        self.beta = disjunction_group[2]

        self.errorFlag = False

        self.tau = 0.0


        self.QDCC = 0.0
        self.qDCC = 0.0
        self.rhoDCC = 0.0

        self.U = 0.0
        self.Lambda = 0.0
        self.negLamN = 0



    def tauDCC(self):

        # (Q,q,rho) previous constraints
        # (a,alpha), (a,beta) branch

        # gets the "tau" value

        invQ = np.linalg.inv(self.Q)

        ua2 = np.transpose(self.a).dot(invQ).dot(self.a)[0,0]
        uq2 = np.transpose(self.q).dot(invQ).dot(self.q)[0,0]
        uauq = np.transpose(self.a).dot(invQ).dot(self.q)[0,0]

        A = (self.alpha-self.beta)**2*ua2
        B = 4*ua2*(uq2-self.rho)-(self.alpha+self.beta+2*uauq)**2+(self.alpha-self.beta)**2
        C = 4*(uq2-self.rho)

        delta = B**2-4*A*C

        if (delta<0):

            print ("tauDCC error: delta<0")
            self.errorFlag = True

        else:

            tau1 = (1.0/(2.0*A)*(-B+np.sqrt(delta)))[0,0]
            tau2 = (1.0/(2.0*A)*(-B-np.sqrt(delta)))[0,0]

            taumin = min(tau1,tau2)
            tau2 = max(tau1,tau2)
            tau1 = taumin

            tauhat = (-1.0/ua2)

            if (tau2<tauhat):

                self.tau = tau2

            elif (tau2==tauhat):

                self.tau = tau1

            elif ((tau1==tau2)&(tau2==tauhat)):

                self.tau = tau1

            else:

                print("tauDCC error: No possible SOCC")
                self.errorFlag = True

    def DCC(self):

        # gets the DCC in the form of x^TQx+2q^Tx+rhp<=0.0 form

        self.tauDCC()

        if (not self.errorFlag):

            self.QDCC = self.Q+self.tau*(self.a).dot(np.transpose(self.a))
            self.qDCC = self.q-self.tau*(self.alpha+self.beta)/2.0*self.a
            self.rhoDCC = self.rho+self.tau*self.alpha*self.beta


    def quad2cone(self):

        # gets DCC in the form of: (x+Q^-1q)^TQ(x+Q^-1q)<=0
        # and then (x+Q^-1q)^T U Lambda U^T (x+Q^-1q)<=0

        self.DCC()

        # get the eigensystem
        [self.Lambda,self.U]=np.linalg.eig(self.QDCC)

        # pick out the single negative eigenvalue
        self.negLamN = self.Lambda.argmin()




    def update_constr(self):

        self.quad2cone()

        invQDCC_q = np.linalg.solve(self.QDCC,self.qDCC)

        x_bar = np.zeros(self.opt_model_N).tolist()
        x_bar_x = np.zeros(self.opt_model_N).tolist()

        j = 0
        for v in self.opt_model.getVars():
            if (j<self.opt_model_N):
                x_bar[j] = v+invQDCC_q[j]
                x_bar_x[j] = v.x+invQDCC_q[j]
                j += 1

        s = np.zeros(self.opt_model_N).tolist()
        for i in range(self.opt_model_N):
            if (i!=self.negLamN):
                s[i] = self.opt_model.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name='s%i' %i)
            else:
                s[i] = self.opt_model.addVar(vtype=GRB.CONTINUOUS,name='s%i' %i)
        self.opt_model.update()

        for i in range(self.opt_model_N):
            loc_U = self.U[:,i]
            loc_sum = 0.0

            for j in range(self.opt_model_N):
                loc_sum += x_bar[j]*loc_U[j]

            if (i==self.negLamN):
                # check positiveness
                loc_sum_2 = 0.0
                for j in range(self.opt_model_N):
                    loc_sum_2 += x_bar_x[j]*loc_U[j]
                if (loc_sum_2<0):
                    loc_sum = -loc_sum

            self.opt_model.addConstr(loc_sum == s[i])

        # LHS<=RHS
        RHS = 0.0
        LHS = 0.0
        for i in range(self.opt_model_N):

            if (i !=self.negLamN):
                LHS += s[i]*s[i]*self.Lambda[i]
            else:
                RHS += s[i]*s[i]*(-self.Lambda[i])
        self.opt_model.addQConstr(LHS<=RHS)









# Create a new model
m = Model("qcp")

# Create variables
N = 3
x = (np.zeros(N)).tolist()
t = m.addVar(vtype=GRB.CONTINUOUS,name='t')
x00 = m.addVar(vtype=GRB.INTEGER,lb = -GRB.INFINITY,name='x00')
for i in range(N):
    x[i] = m.addVar(vtype=GRB.CONTINUOUS,lb = -GRB.INFINITY,name='x%i' %i)

# Integrate new variables
m.update()

# Set objective: x
obj = 0.0
obj = x[0]
m.setObjective(obj, GRB.MAXIMIZE)

# set constraitns
lin_LHS = 0.0
lin_LHS += 2.0*t+x00
for i in range(N):
    lin_LHS += x[i]
m.addConstr(lin_LHS==14.0/3.0-4.0/3.0-1.0)

quad_LHS = 0.0
quad_LHS += x00*x00
for i in range(N):
    quad_LHS += x[i]*x[i]
m.addQConstr(quad_LHS<=t*t)

# optimize
default_start = time.time()
m.optimize()
default_end = time.time()
default_duration = default_end-default_start

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))
print('default_duration',default_duration)










#
A = np.matrix(np.zeros((1,N+2)))
for i in range(N+2):
    A[0,i] = 1.0
A[0,0] = 2.0

b = np.matrix([[14.0/3.0-4.0/3.0-1.0]])

J = np.matrix(np.zeros((N+2,N+2)))
for i in range(N+2):
    J[i,i] = 1.0
J[0,0] = -1.0

c = np.matrix(np.zeros((N+2,1)))
c[2] = 1.0


model = SOCP_DCC(A,b,J,c,1,[-2.0,-1.0])


model.Nullspace_DCC()
