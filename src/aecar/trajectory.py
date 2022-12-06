
__all__ = ["Trajectory"]

import numpy as np
from typing import TYPE_CHECKING, List
import matplotlib.pyplot as plt
import os, sys
from .path import Path
from .properties import Properties
from .initial_state import InitialState

class CostSettings:
    def __init__(
        self,
        position=10.0,
        speed=5.0,
        turn_angle=10.0,
        speed_control=3.0,
        turn_control=3.0,
        gamma = 0.9,
    ):
        self._position = position
        self._speed = speed
        self._turn_angle = turn_angle
        self._speed_control = speed_control
        self._turn_control = turn_control
        self._gamma = gamma

    @property
    def position(self) -> float:
        return self._position

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def turn_angle(self) -> float:
        return self._turn_angle

    @property
    def speed_control(self) -> float:
        return self._speed_control

    @property
    def turn_control(self) -> float:
        return self._turn_control

    @property
    def gamma(self) -> float:
        return self._gamma

class Trajectory:
    """
    Car trajectory object stores controls and full car states
    """
    def __init__(
        self,
        path:Path,
        cost_settings:CostSettings=None,
        initial_state:InitialState=None,
        properties:Properties=None,
    ):
        self._path = path
        self._cost_settings = cost_settings if cost_settings is not None else CostSettings()
        self._initial_state = initial_state if initial_state is not None else path.initial_state
        self._properties = properties if properties is not None else Properties()

        # initialize variables
        self._initialize_variables()

        # setup a save folder
        self._save_folder = os.path.join(os.getcwd(), f"{self.path.name}")
        if not os.path.exists(self._save_folder):
            os.mkdir(self._save_folder)

        # initialize the trajectory
        self.update()

    def _initialize_variables(self):
        """
        initialize all the variables and arrays for trajectory optimization
        """
        self._x = np.zeros((5,self.n_time))
        self._u = np.zeros((2,self.n_time-1))
        self._costs = np.zeros((self.n_time))

        self._track_error = np.zeros((5,self.n_time))
        self._quad_track_error = np.zeros((15,self.n_time))

        # inititalize tracking error state gradients
        self._dtrack_dspeed = np.zeros((5,self.n_time))
        self._dtrack_dturn = np.zeros((5,self.n_time))
        self._dquad_dspeed = np.zeros((15,self.n_time))
        self._dquad_dturn = np.zeros((15,self.n_time))

        # initial kernel matrix
        self._kernel = np.zeros((15))

        self._kernel_history = []
        self._kernel_history.append([self.kernel[j] for j in range(15)])

    @property
    def path(self) -> Path:
        return self._path

    @property
    def cost_settings(self) -> CostSettings:
        return self._cost_settings

    @property
    def initial_state(self) -> InitialState:
        return self._initial_state

    @property
    def properties(self) -> Properties:
        return self._properties

    @property
    def n_time(self) -> int:
        return self.path.n_time
    
    @property
    def time(self) -> List[float]:
        return self.path.time

    @property
    def save_folder(self):
        return self._save_folder

    @property
    def state(self) -> List[float]:
        return self._x

    @property
    def x1(self) -> List[float]:
        return self.state[0,:]

    @x1.setter
    def x1(self, new_x1):
        self._x[0,:] = new_x1

    @property
    def x2(self) -> List[float]:
        return self.state[1,:]

    @x2.setter
    def x2(self, new_x2):
        self._x[1,:] = new_x2
        
    @property
    def speed(self) -> List[float]:
        return self.state[2,:]

    @property
    def theta(self) -> List[float]:
        return self.state[3,:]

    @property
    def x1_speed(self) -> List[float]:
        return self.speed * np.cos(self.theta)

    @property
    def x2_speed(self) -> List[float]:
        return self.speed * np.sin(self.theta)

    @property
    def turn_angle(self) -> List[float]:
        return self.state[4,:]

    @property
    def control(self) -> List[float]:
        return self._u

    @control.setter
    def control(self, new_control):
        self._u = new_control

    @property
    def speed_control(self) -> List[float]:
        return self.control[0,:]

    @property
    def turn_control(self) -> List[float]:
        return self.control[1,:]
    
    @property
    def track_error(self) -> List[float]:
        return self._track_error

    @property
    def quad_track_error(self) -> List[float]:
        return self._quad_track_error

    @property
    def costs(self) -> List[float]:
        return self._costs

    @property
    def kernel(self) -> List[float]:
        return self._kernel

    def update(self):
        """
        update the full trajectory using the nonlinear car dynamics
        """

        # update the states for the given controls
        self._x[:,0] = self.initial_state.state
        for i in range(self.n_time-1):
            dt = self.time[i+1] - self.time[i]
            self._x[0,i+1] = self._x[0,i] + dt * self.x1_speed[i]
            self._x[1,i+1] = self._x[1,i] + dt * self.x2_speed[i]
            self._x[2,i+1] = self._x[2,i] + dt * self.speed_control[i] / self.properties.mass
            self._x[3,i+1] = self._x[3,i] + dt * self.speed[i] * self.turn_angle[i] / self.properties.length
            self._x[4,i+1] = self._x[4,i] + dt * self.turn_control[i] / self.properties.inertia

        # update the path tracking error
        self._track_error[0,:] = self.x1 - self.path.x1
        self._track_error[1,:] = self.x2 - self.path.x2
        self._track_error[2,:] = self.x1_speed - self.path.x1_dot
        self._track_error[3,:] = self.x2_speed - self.path.x2_dot
        self.track_error[4,:] = (
            self.path.x2_ddot * np.cos(self.theta) - \
            self.path.x1_ddot * np.sin(self.theta)
        ) - self.speed * self.speed * self.turn_angle / self.properties.length

        #print(f"track error = {self._track_error}")

        # compute path tracking derivatives w.r.t. speed
        self._dtrack_dspeed[2,:] = np.cos(self.theta)
        self._dtrack_dspeed[3,:] = np.sin(self.theta)
        self._dtrack_dspeed[4,:] = -2.0 * self.speed * self.turn_angle / self.properties.length
        # compute path tracking derivatives w.r.t. turn angle
        self._dtrack_dturn[4,:] = -1.0 * self.speed * self.speed / self.properties.length 

        # update the quadratic path tracking error and derivatives
        ct = 0
        for i in range(5):
            for j in range(5):
                if i <= j:
                    self._quad_track_error[ct,:] = self.track_error[i,:] * self.track_error[j,:]
                    #print(f"quad track error {ct} = {self._quad_track_error[i,:]}")

                    # compute speed derivative of quad path tracking error w/ product rule
                    self._dquad_dspeed[ct,:] = self._dtrack_dspeed[i,:] * self.track_error[j,:] +\
                        self.track_error[i,:] * self._dtrack_dspeed[j,:]

                    # compute turn angle derivative of quad path tracking error w/ product rule
                    self._dquad_dturn[ct,:] = self._dtrack_dturn[i,:] * self.track_error[j,:] +\
                        self.track_error[i,:] * self._dtrack_dturn[j,:]

                    ct += 1

        #print(f"final ct = {ct}")
        #print(f"quad track error = {self._quad_track_error}")

        # update the cost functionals
        for i in range(self.n_time):
            self._costs[i] = self.cost_settings.position * self._track_error[0,i]**2 + \
                self.cost_settings.position * self._track_error[1,i]**2 + \
                self.cost_settings.speed * self._track_error[2,i]**2 + \
                self.cost_settings.speed * self._track_error[3,i]**2 + \
                self.cost_settings.turn_angle * self._track_error[4,i]**2

            # controls cost
            if i < self.n_time - 1:
                self._costs[i] += self.cost_settings.speed_control * self.speed_control[i]**2 + \
                    self.cost_settings.turn_control * self.turn_control[i]**2

        return

    def update_controls(self, start_index):
        """
        update the optimal controls of the kernel matrix
        """
        # update the controls using the kernel matrix
        for i in range(start_index,self.n_time-1):
            dt = self.time[i+1] - self.time[i]

            # compute derivatives of kernel cost at state i+1
            dcost_dspeed = sum(self.kernel * self._dquad_dspeed[:,i+1])
            dcost_dturn = sum(self.kernel * self._dquad_dturn[:,i+1])

            #print(f"dquad/dspeed = {self._dquad_dspeed[:,i+1]}")
            #print(f"dquad/dturn = {self._dquad_dturn[:,i+1]}")

            # optimal speed control
            self._u[0,i] = -0.5 / self.cost_settings.speed_control * dt / self.properties.mass * dcost_dspeed
            self._u[1,i] = -0.5 / self.cost_settings.turn_control * dt / self.properties.inertia * dcost_dturn

    def train_online(self, width=10, plot=False):

        start_index = 0
        final_index = min(width,self.n_time-1) - 1

        while final_index < self.n_time - 1:
            indices = [i for i in range(start_index,final_index+1)]

            # update controls with current kernel and then train it
            self.update_controls(0)
            self.update()
            self.train_kernel(indices)
            if plot: self.plot_path()

            # update indices bounds for next run
            if final_index == self.n_time - 2:
                break
            start_index=final_index+1
            final_index=width-1 + start_index
            final_index = min(final_index, self.n_time-2)


    def train_kernel(self, indices:List[float]):
        """
        train the kernel matrix W with recursive least-squares
        value function = sum(kernel*quad_tracking_error)

        Y = X^T w to solve for w uses
        XY = (XX^T) w
        w_LS = inv(XX^T) X Y

        Training occurs over the time indices given by indices
        """
        
        # track error matrix w shape (15,indices) and tp is (indices,15)
        track_error_matrix = self.quad_track_error[:,indices]
        track_error_matrix_tp = np.transpose(self.quad_track_error[:,indices])

        # compute X*X^T matrix of shape (15,15)
        XXT = np.matmul(track_error_matrix, track_error_matrix_tp)

        # determine the Y training vector
        Y = np.zeros((len(indices), 1))
        for i,index in enumerate(indices):
            Y[i,0] = self.costs[index] + sum(self.kernel * self.quad_track_error[:,index+1])

        # compute XY product of shape (15,1)
        XY = np.matmul(track_error_matrix, Y)

        # condition the matrix
        sigma = 1e-10
        for id in range(15):
            XXT[id,id] += sigma

        # solve the matrix equation (XX^T) * w_LS = XY
        kernel_LS = np.linalg.solve(XXT, XY)

        # update the kernel
        self._kernel[:] = kernel_LS[:,0]

        # test the residual of least squares
        XTw = np.matmul(track_error_matrix_tp,kernel_LS)
        resid = np.reshape(XTw - Y, (len(indices)))

        print(f"LS resid = {resid}",flush=True)
        print(f"kernel LS = {self.kernel}")
        self._kernel_history.append([self.kernel[j] for j in range(15)])

    def filepath(self, filename):
        return os.path.join(self.save_folder, filename)

    def plot_path(self, save=True):
        """
        plot the xy path taken in space
        """
        plt.figure()
        plt.plot(self.path.x1,self.path.x2, "b--", linewidth=3, label="path")
        plt.plot(self.x1, self.x2, "k-", linewidth=2, label="traj")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        if save:
            plt.savefig(self.filepath("path.png"))
        else:
            plt.show()
        plt.close()

    def plot_states(self, save=True):
        """
        plot the trajectory states
        """
        plt.figure()
        plt.plot(self.time, self.x1, "k-", linewidth=2, label="x1")
        plt.plot(self.time,self.x2, "b-", linewidth=2, label="x2")
        plt.plot(self.time,self.speed, "c-", linewidth=2, label="speed")
        plt.plot(self.time,self.theta, "g-", linewidth=2, label="theta")
        plt.plot(self.time,self.turn_angle, "r-", linewidth=2, label="phi")
        plt.xlabel("time")
        plt.ylabel("states")
        plt.legend()
        if save:
            plt.savefig(self.filepath("states.png"))
        else:
            plt.show()
        plt.close()


    def plot_kernel(self, save=True):
        """
        plot the kernel history
        """
        nhist = len(self._kernel_history)
        kernel_mat = np.zeros((15,nhist))
        iterations = [j for j in range(nhist)]
        for ihist in range(nhist):
            kernel = self._kernel_history[ihist]
            kernel_mat[:,ihist] = kernel[:]

        colors = "kbcgr"
        plt.figure()
        for i in range(15):
            plt.plot(iterations, kernel_mat[i,:], colors[i%5] + "-", linewidth=2, label=f"kernel{i}")
        plt.xlabel("iterations")
        plt.ylabel("kernel entries")
        #plt.legend()
        if save:
            plt.savefig(self.filepath("kernel.png"))
        else:
            plt.show()
        plt.close()

    def plot_error(self, save=True):
        """
        plot the tracking error
        """
        colors = "kbcgr"
        plt.figure()
        for i in range(5):
            plt.plot(self.time, self.track_error[i,:], colors[i] + "-", linewidth=2, label=f"e{i}")
        plt.xlabel("time")
        plt.ylabel("track error")
        plt.legend()
        if save:
            plt.savefig(self.filepath("tracking_error.png"))
        else:
            plt.show()
        plt.close()