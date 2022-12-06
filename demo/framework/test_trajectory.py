import aecar
import numpy as np

# test each of the built-in paths
rose_path = aecar.Path.rose()
init_rose = rose_path.initial_state
init_rose.theta = np.pi/2
rose_traj = aecar.Trajectory(rose_path, initial_state=init_rose)

rose_traj.plot_path()
rose_traj.plot_states()