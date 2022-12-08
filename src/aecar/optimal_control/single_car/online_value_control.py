
__all__ = ["OnlineValueControl"]

import numpy as np
import matplotlib.pyplot as plt
import os
from aecar.car import Car
from typing import TYPE_CHECKING, List

class OnlineValueControl:
    """
    Use least-squares kernel approximation and value gradients to learn the optimal controls
    """

    def __init__(
        self,
        car:Car
    ):
        self._car = car
        self._initialize_kernel()

    @property
    def car(self) -> Car:
        return self._car

    @property
    def trajectory(self):
        return self.car.trajectory

    @property
    def cost_settings(self):
        return self.trajectory.cost_settings

    @property
    def properties(self):
        return self.car.properties

    @property
    def path(self):
        return self.car.path

    @property
    def n_time(self) -> int:
        return self.trajectory.n_time

    @property
    def time(self):
        return self.trajectory.time

    @property
    def kernel(self):
        return self._kernel

    @property
    def save_folder(self):
        return self.trajectory.save_folder

    @property
    def costs(self):
        return self.trajectory.costs

    def _initialize_kernel(self):

        # initial kernel matrix
        self._kernel = np.zeros((15))

        self._kernel_history = []
        self._kernel_history.append([self.kernel[j] for j in range(15)])

    def update_controls(self, start_index):
        """
        update the optimal controls of the kernel matrix
        """
        # update the controls using the kernel matrix
        for i in range(start_index,self.n_time-1):
            dt = self.time[i+1] - self.time[i]

            # compute derivatives of kernel cost at state i+1
            dcost_dspeed = sum(self.kernel * self.trajectory._dquad_dspeed[:,i+1])
            dcost_dturn = sum(self.kernel * self.trajectory._dquad_dturn[:,i+1])

            #print(f"dquad/dspeed = {self._dquad_dspeed[:,i+1]}")
            #print(f"dquad/dturn = {self._dquad_dturn[:,i+1]}")

            # optimal speed control
            self.trajectory.speed_control[i] = -0.5 / self.cost_settings.speed_control * dt / self.properties.mass * dcost_dspeed
            self.trajectory.turn_control[i] = -0.5 / self.cost_settings.turn_control * dt / self.properties.inertia * dcost_dturn

    def train_online(self, width=10, plot=False):

        start_index = 0
        final_index = min(width,self.n_time-1) - 1

        while final_index < self.n_time - 1:
            indices = [i for i in range(start_index,final_index+1)]

            # update controls with current kernel and then train it
            self.update_controls(0)
            self.trajectory.update()
            self.train_kernel(indices)
            if plot: self.trajectory.plot_path()

            # update indices bounds for next run
            if final_index == self.n_time - 2:
                break
            start_index=final_index+1
            final_index=width-1 + start_index
            final_index = min(final_index, self.n_time-2)

        self.post_process()


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
        track_error_matrix = self.trajectory.quad_track_error[:,indices]
        track_error_matrix_tp = np.transpose(self.trajectory.quad_track_error[:,indices])

        # compute X*X^T matrix of shape (15,15)
        XXT = np.matmul(track_error_matrix, track_error_matrix_tp)

        # determine the Y training vector
        Y = np.zeros((len(indices), 1))
        for i,index in enumerate(indices):
            Y[i,0] = self.costs[index] + sum(self.kernel * self.trajectory.quad_track_error[:,index+1])

        # compute XY product of shape (15,1)
        XY = np.matmul(track_error_matrix, Y)

        # condition the matrix
        sigma = 1e-5
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
        return self.trajectory.filepath(filename)

    def post_process(self):
        
        self.plot_kernel()

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