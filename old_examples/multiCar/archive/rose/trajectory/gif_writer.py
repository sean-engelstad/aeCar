import utils
from utils.gifs import GifWriter
import os

my_writer = GifWriter(frames_per_second=10)
my_writer(gif_filename="mrose_traj.gif", path=os.getcwd())