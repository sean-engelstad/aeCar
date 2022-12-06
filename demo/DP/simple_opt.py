# import packages
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize

# choose a trajectory, in this case a rose
n = 20
t = np.linspace(0,10,n)
dt = t[1] - t[0]
x1 = np.reshape(2 * np.cos(2*t) * np.cos(t), (n,1))
x2 = np.reshape(2 * np.cos(2*t) * np.sin(t), (n,1))
xdes = np.concatenate((x1,x2),axis=1)
#plt.plot(x1,x2)
#plt.show()

# problem settings
x0 = [2,0,0.01,np.pi/2,0] #x1,x2,S,theta,phi
L = 1; m = 1; I = 1

# select an initial control policy
u = np.zeros((n-1,2), dtype=complex) * 0.01

# define the state trajectory function
def trajectory(xi,uc):
    x = np.zeros((n,5), dtype=complex)
    x[0,:] = xi
    for i in range(n-1):
        x[i+1,0] = x[i,0] + dt * x[i,2] * np.cos(x[i,3])
        x[i+1,1] = x[i,1] + dt * x[i,2] * np.sin(x[i,3])
        x[i+1,2] = x[i,2] + dt * uc[i,0] / m
        x[i+1,3] = x[i,3] + dt * x[i,2] * x[i,4] / L
        x[i+1,4] = x[i,4] + dt * uc[i,1] / I
    return x

# compute the state trajectory
xact = trajectory(x0,u)
def plot_trajectory(xc,xd,final=n):
    plt.plot(xd[:final,0],xd[:final,1],"b--",label="desired")
    x1 = [v.real for v in xc[:final,0]]
    x2 = [v.real for v in xc[:final,1]]
    plt.plot(x1,x2,"k-",label="current")
    
    plt.legend()
    plt.show()
#plot_trajectory(xact,xdes)

# define the path cost
gamma = 0.999
def value_function(xc,xd,uc):
    cost = 0.0
    x = np.copy(xc)
    for i in range(n-1):
        des_speed = np.sqrt((xd[i+1,0]-xd[i,0])**2 + (xd[i+1,1]-xd[i,1])**2)/dt
        des_angle = np.arctan2(xd[i+1,0]-xd[i,0],xd[i+1,1]-xd[i,1])
        cost += (10 * (x[i,0]-xd[i,0])**2 + 10 * (x[i,1] - xd[i,1])**2 + 3 * (x[i,2]-des_speed)**2 + 3 * (x[i,3] - des_angle)**2)
        cost += (0.02 * (uc[i,0]**2 + uc[i,1]**2))

        # compute next state
        x[i+1,0] = x[i,0] + dt * x[i,2] * np.cos(x[i,3])
        x[i+1,1] = x[i,1] + dt * x[i,2] * np.sin(x[i,3])
        x[i+1,2] = x[i,2] + dt * uc[i,0] / m
        x[i+1,3] = x[i,3] + dt * x[i,2] * x[i,4] / L
        x[i+1,4] = x[i,4] + dt * uc[i,1] / I

    # add final state cost
    cost += 5 * (x[n-1,0]-xd[n-1,0])**2 + 5 * (x[n-1,1] - xd[n-1,1])**2
    return cost

def value_fn(uin):
    uin2 = np.reshape(uin,(n-1,2))
    cost = value_function(xact,xdes,uin2)
    if sum(uin).imag == 0.0:
        print(f"cost = {cost}")
    return cost

def value_gradient(uc):
    grad = np.zeros((2*(n-1)))
    eps = 1e-30
    utemp = np.zeros((2*(n-1)), dtype=complex)
    for i in range(2*(n-1)):
        utemp[:] = uc[:]
        utemp[i] += 1j * eps
        grad[i] = value_fn(utemp).imag/eps
    #print(f"utemp = {utemp}")
    return grad

u0 = np.reshape(u, (2*(n-1)))

print(f"val grad = {value_gradient(u0)}")

res = minimize(value_fn, u0, method='BFGS', jac=value_gradient,
               options={'disp': True})

uopt = res.x
uopt2 = np.reshape(uopt, (n-1,2))

xopt = trajectory(x0,uopt2)
plot_trajectory(xopt,xdes)