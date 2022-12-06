# import packages
import numpy as np
import matplotlib.pyplot as plt
import os, sys


# choose a trajectory, in this case a rose
class CarManager:
    def __init__(self,
    name="",
    x0=[0,0,0,0,0],
    x1_fn=None,
    x2_fn=None,
    n=10, 
    n_fixed=0,
    QP=100, 
    QS=5, 
    R=0.2,
    tf_100=10.0,
    ):
        self.name = name
        self._n = n
        self.n_fixed = n_fixed
        self.QP = QP
        self.QPF = QP
        self.QS = QS
        self.R = R

        self.x0 = x0
        self.x1_fn = x1_fn
        self.x2_fn = x2_fn
        self.tf_100 = tf_100

        # call initialize variables
        self._initialize_variables()

        # make super folder
        self.super_folder = os.path.join(os.getcwd(), name)
        if not os.path.exists(self.super_folder):
            os.mkdir(self.super_folder)

        # make a folder to save the runs in
        self.trajectory_folder = os.path.join(self.super_folder, "trajectory")
        if not os.path.exists(self.trajectory_folder):
            os.mkdir(self.trajectory_folder)

        # make a folder to save the runs in
        # self.trajectory_folder = os.path.join(self.super_folder, "trajectory")
        # if not os.path.exists(self.trajectory_folder):
        #     os.mkdir(self.trajectory_folder)

        self.state_folder = os.path.join(self.super_folder, "states")
        if not os.path.exists(self.state_folder):
            os.mkdir(self.state_folder)
        

    @property
    def n(self) -> float:
        return self._n

    @n.setter
    def n(self, new_n):
        self.n = new_n
        self._initialize_variables()
        

    def _initialize_variables(self):
        # problem settings
        #self.x0 = [2,0,0.5,np.pi/2,0.1] #x1,x2,S,theta,phi
        self.L = 1
        self.m = 1
        self.I = 1

        # select an initial control policy
        #u = np.zeros((self.n-1,2), dtype=complex)

        # compute the desired path
        self._desired_path()

    def _desired_path(self):
        self.t = np.linspace(0,self.n*self.tf_100/100,self.n)
        self.dt = self.t[1] - self.t[0]
        x1 = np.reshape(self.x1_fn(self.t), (self.n,1))
        x2 = np.reshape(self.x2_fn(self.t), (self.n,1))
        self.xdes = np.concatenate((x1,x2),axis=1)

    def trajectory(self,uc):
        x = np.zeros((self.n,5), dtype=complex)
        x[0,:] = self.x0
        for i in range(self.n-1):
            x[i+1,0] = x[i,0] + self.dt * x[i,2] * np.cos(x[i,3])
            x[i+1,1] = x[i,1] + self.dt * x[i,2] * np.sin(x[i,3])
            x[i+1,2] = x[i,2] + self.dt * uc[i,0] / self.m
            x[i+1,3] = x[i,3] + self.dt * x[i,2] * x[i,4] / self.L
            x[i+1,4] = x[i,4] + self.dt * uc[i,1] / self.I
        return x

    def plot_trajectory(self,uc,final=None):
        if final is None: final = self.n
        xc = self.trajectory(uc)
        plt.figure()
        plt.plot(self.xdes[:final,0],self.xdes[:final,1],"b--",label="desired")
        x1 = [v.real for v in xc[:final,0]]
        x2 = [v.real for v in xc[:final,1]]
        plt.plot(x1,x2,"k-",label="actual")
        
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        #plt.show()
        plt.savefig(os.path.join(self.trajectory_folder, f"traj_{self.n}.png"))
        plt.close()

    def plot_states(self,uc,final=None):
        if final is None: final = self.n
        xc = self.trajectory(uc)
        
        x1 = [v.real for v in xc[:final,0]]
        x2 = [v.real for v in xc[:final,1]]
        V = [v.real for v in xc[:final,2]]
        theta = [v.real % (2 * np.pi) for v in xc[:final,3]]
        phi = [v.real for v in xc[:final,4]]
        plt.figure()
        plt.plot(self.t,x1,"k-",label="x")
        plt.plot(self.t,x2,"b-",label="y")
        plt.plot(self.t,V,"c-",label="speed")
        plt.plot(self.t,theta,"g-",label="angle")
        plt.plot(self.t, phi, "r-", label="turn_angle")
        
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("state")
        #plt.show()
        plt.savefig(os.path.join(self.state_folder, f"state_{self.n}.png"))
        plt.close()

    def value_function(self,uc):
        cost = 0.0
        #x = np.copy(self.xc)
        x = np.zeros((self.n,5), dtype=complex)
        x[0,:] = self.x0
        for i in range(self.n-1):
            xspeed = (self.xdes[i+1,0] - self.xdes[i,0])/self.dt
            yspeed = (self.xdes[i+1,1] - self.xdes[i,1])/self.dt

            cost += (self.QP * (x[i,0]-self.xdes[i,0])**2 + self.QP * (x[i,1] - self.xdes[i,1])**2 +\
                self.QS * (x[i,2]*np.cos(x[i,3])-xspeed)**2 + self.QS * (x[i,2] * np.sin(x[i,3]) - yspeed)**2 )
            cost += (self.R * (uc[i,0]**2 + uc[i,1]**2))

            # compute next state
            x[i+1,0] = x[i,0] + self.dt * x[i,2] * np.cos(x[i,3])
            x[i+1,1] = x[i,1] + self.dt * x[i,2] * np.sin(x[i,3])
            x[i+1,2] = x[i,2] + self.dt * uc[i,0] / self.m
            x[i+1,3] = x[i,3] + self.dt * x[i,2] * x[i,4] / self.L
            x[i+1,4] = x[i,4] + self.dt * uc[i,1] / self.I

        # add final state cost
        cost += self.QPF * (x[self.n-1,0]-self.xdes[self.n-1,0])**2 + self.QPF * (x[self.n-1,1] - self.xdes[self.n-1,1])**2
        return cost

    def analysis(self,udict):
        uvec = np.zeros((2*(self.n-1)), dtype=complex)
        ct = 0
        for ukey in udict:
            uvec[ct] = float(udict[ukey])
            ct += 1
        uin2 = np.reshape(uvec,(self.n-1,2))
        cost = self.value_function(uin2)
        if sum(uvec).imag == 0.0:
            print(f"cost = {cost}")
        funcs = {"obj" : cost}
        return funcs

    def gradient(self,udict,funcs):
        grad = np.zeros((2*(self.n-1)))
        eps = 1e-30
        uinput = np.zeros((2*(self.n-1)), dtype=complex)
        ct = 0
        for ukey in udict:
            uinput[ct] = float(udict[ukey])
            ct += 1
        utemp = np.zeros((2*(self.n-1)), dtype=complex)
        for i in range(2*(self.n-1)):
            if i < 2*(self.n_fixed-1):
                grad[i] = 0.0
            else:
                utemp[:] = uinput[:]
                utemp[i] += 1j * eps
                uin2 = np.reshape(utemp,(self.n-1,2))
                grad[i] = self.value_function(uin2).imag/eps
        sens = {}
        sens["obj"] = {}
        ct = 0
        for ukey in udict:
            sens["obj"][ukey] = grad[ct]
            ct += 1
        #print(f"utemp = {utemp}")
        return sens