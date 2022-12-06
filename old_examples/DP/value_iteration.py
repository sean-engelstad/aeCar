# import packages
import numpy as np
import matplotlib.pyplot as plt
import sys

# choose a trajectory, in this case a rose
n = 10
t = np.linspace(0,0.2,n)
dt = t[1] - t[0]
x1 = np.reshape(2 * np.cos(2*t) * np.cos(t), (n,1))
x2 = np.reshape(2 * np.cos(2*t) * np.sin(t), (n,1))
xdes = np.concatenate((x1,x2),axis=1)
#plt.plot(x1,x2)
#plt.show()

# problem settings
x0 = [2.0,0,0.5,np.pi/2,1.0] #x1,x2,S,theta,phi
L = 1; m = 1; I = 1

# select an initial control policy
u = np.ones((n-1,2), dtype=complex) * 0.01

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
plot_trajectory(xact,xdes)

# define the path cost
gamma = 0.999
R = 0.01
def value_function(xc,xd,uc,index,final=n-1):
    cost = 0.0
    x = np.copy(xc)
    #print(f"x = {x}")
    #print(f"x index = {x[index,:]}")
    for i in range(index,final):
        des_speed = np.sqrt((xd[i+1,0]-xd[i,0])**2 + (xd[i+1,1]-xd[i,1])**2)
        des_angle = np.arctan2(xd[i+1,0]-xd[i,0],xd[i+1,1]-xd[i,1])
        cost += (10 * (x[i,0]-xd[i,0])**2 + 10 * (x[i,1] - xd[i,1])**2 + 3 * (x[i,2]-des_speed)**2 + 3 * (x[i,3] - des_angle)**2)
        cost += (0.5 * R * (uc[i,0]**2 + uc[i,1]**2))

        #print(f"S imag part at i = {x[i,2].imag}")
        # compute next state
        x[i+1,0] = x[i,0] + dt * x[i,2] * np.cos(x[i,3])
        x[i+1,1] = x[i,1] + dt * x[i,2] * np.sin(x[i,3])
        x[i+1,2] = x[i,2] + dt * uc[i,0] / m
        x[i+1,3] = x[i,3] + dt * x[i,2] * x[i,4] / L
        x[i+1,4] = x[i,4] + dt * uc[i,1] / I

        #print(f"phi imag part at i = {x[i,4].imag}")

    # add final state cost
    #print( final = {x[final-1,1]}")f"x
    cost += 5 * (x[final-1,0]-xd[final-1,0])**2 + 5 * (x[final-1,1] - xd[final-1,1])**2
    #print(f"cost = {cost}")
    return cost

def dVdS(xc,xd,uc,index,final=n-1):
    # copy current trajectory but add perturbation to S state
    xc2 = np.copy(xc)
    eps = 1e-10
    xc2[index,2] += 1j * eps
    #print(f"xc2 perturbed = {xc2[index-1,2]}")
    return value_function(xc2,xd,uc,index,final).imag/eps

def dVdphi(xc,xd,uc,index,final=n-1):
    # copy current trajectory but add perturbation to S state
    xc2 = np.copy(xc)
    eps = 1e-10
    xc2[index,4] += 1j * eps
    return value_function(xc2,xd,uc,index,final).imag/eps

def dVduS(xc,xd,uc,index,final=n-1):
    # copy current trajectory but add perturbation to S state
    uc2 = np.copy(uc)
    eps = 1e-10
    uc2[index,0] += 1j * eps
    return value_function(xc,xd,uc2,index,final).imag/eps

def dVduphi(xc,xd,uc,index,final=n-1):
    # copy current trajectory but add perturbation to S state
    uc2 = np.copy(uc)
    eps = 1e-10
    uc2[index,1] += 1j * eps
    return value_function(xc,xd,uc2,index,final).imag/eps

# compute dVdS at state one
def deriv_demo():
    c_dvds = dVdS(xact,xdes,u,index=1)
    dVdphi_init = dVdphi(xact,xdes,u,index=1)
    mindu = 100
    for newu in np.linspace(-1.0,5.0,100):
        #u[0,0] = -1.0/R * dt/m * c_dvds
        u[0,0] = newu
        dVduS_init = dVduS(xact, xdes, u, index=0)
        if abs(dVduS_init) < mindu:
            mindu = abs(dVduS_init)
            minu = newu
    print(f"minu = {minu}")
    print(f"mindu = {mindu}")
    origu = -1.0/R * dt/m * c_dvds
    uratio = minu / origu
    print(f"origu = {origu}")
    print(f"u ratio = {uratio}")
    
    print(f"dVdS = {c_dvds}")
    print(f"dVdphi = {dVdphi_init}")
    print(f"dt = {dt}")
    print(f"dVduS = {dVduS_init}")
#deriv_demo()
# turns out dVdS is a function of u also so that's why it doesn't work well before?
#sys.exit()

# check dVdu 

# perform value iteration on the control law
# ---------------------------------------------

# min and max controls
umin = -100
umax = 100

# backwards loop through the stages changing the controls
uprev = np.copy(u)
for irep in range(10):
    for k in range(n-1):
        cost = value_function(xact,xdes,uprev,index=k)
        c_dvds = dVdS(xact,xdes,uprev,index=k+1)
        c_dvdphi = dVdphi(xact,xdes,uprev,index=k+1)
        #print(f"dVdS at {k+1} = {c_dvds}")
        #print(f"dVdphi at {k+1}= {c_dvdphi}")
        u[k,0] = -1.0/R * dt/m * c_dvds
        u[k,1] = -1.0/R * dt/I *  c_dvdphi
        #u[k,0] = min(max(u[k,0],umin),umax)
        #u[k,1] = min(max(u[k,1],umin),umax)
        cost2 = value_function(xact,xdes,u,index=k)
        dcost = cost2 - cost
        print(f"delta cost = {dcost}")
        #print(f"u0 at {k} = {u[k,0]}")
        #print(f"u1 at {k} = {u[k,1]}")
        uprev = u
        
        xact = trajectory(x0,u)
    # plot final trajectory
    
    cost = value_function(xact,xdes,uprev,index=k)
    #print(f"cost = {cost}")
    
# uprev = np.copy(u)
# alpha = 100
# for irep in range(5):
#     for k in range(n-1):
#         # loop to minimize current cost function with respect to u
#         for irep in range(10):
#             cost = value_function(xact,xdes,uprev,index=k)
#             u[k,0] -= alpha * dVduS(xact,xdes,uprev,index=k)
#             u[k,1] -= alpha * dVduphi(xact,xdes,uprev,index=k)
#             cost2 = value_function(xact,xdes,u,index=k)
#             print(f"dphi = {dVduphi(xact,xdes,uprev,index=k)}")
#             print(f"delta cost = {cost2 - cost}")
#             uprev = u
        
#     # plot final trajectory
    
#     cost = value_function(xact,xdes,uprev,index=k)
#     print(f"cost = {cost}")

# xact = trajectory(x0,u)
plot_trajectory(xact,xdes)

