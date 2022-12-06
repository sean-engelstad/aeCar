import aecar
import numpy as np

# test each of the built-in paths
rose_path = aecar.Path.parabola(time_vec=np.linspace(0.0,10.0,3000))
rose_traj = aecar.Trajectory(rose_path)

rose_traj.plot_error()
rose_traj.train_online(width=100)
rose_traj.plot_path()