import aecar
import numpy as np

circle_path = aecar.Path.circle()
circle_traj = aecar.Trajectory(path=circle_path)
circle_traj.initial_state.turn_angle = 1.0
circle_traj.initial_state.speed = 1.0
circle_traj.x1 = circle_path.x1
circle_traj.x2 = circle_path.x2
circle_traj.update()
circle_traj.plot_path()
circle_traj.plot_error()