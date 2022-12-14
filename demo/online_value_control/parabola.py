import aecar
import numpy as np

# test each of the built-in paths
path = aecar.Path.parabola(time_vec=np.linspace(0.0,40.0,3000))
traj = aecar.Trajectory(path)

traj.plot_error()
traj.train_online(width=200, plot=True)